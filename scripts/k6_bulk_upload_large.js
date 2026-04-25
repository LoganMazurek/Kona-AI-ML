import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

const BASE_URL = (__ENV.BASE_URL || 'https://loganmazurek.com').replace(/\/$/, '');
const LOGIN_ID = __ENV.LOGIN_ID || '';
const LOGIN_PASSWORD = __ENV.LOGIN_PASSWORD || '';
const BULK_FILE = __ENV.BULK_FILE || 'outputs/manual_batch_test_100_rows.xlsx';
const PEAK_VUS = Number(__ENV.PEAK_VUS || '25');
const PRE_PEAK_VUS = Math.max(10, Math.floor(PEAK_VUS * 0.8));
const BULK_P95_MS = Number(__ENV.BULK_P95_MS || '12000');
const BULK_P99_MS = Number(__ENV.BULK_P99_MS || '20000');

const bulkFixtureBytes = open(BULK_FILE, 'b');

const bulkUploadDuration = new Trend('bulk_upload_duration', true);
const loginFailureRate = new Rate('login_failures');
const bulkFailureRate = new Rate('bulk_upload_failures');
const bulkHttpErrors = new Counter('bulk_upload_http_errors');
const bulkServerErrors = new Counter('bulk_upload_server_errors');

let loggedIn = false;
let sessionToken = '';
let bulkFailureSamples = 0;
const MAX_BULK_FAILURE_SAMPLES = 5;

export const options = {
  scenarios: {
    bulk_upload_load: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '2m', target: 3 },
        { duration: '3m', target: 8 },
        { duration: '4m', target: PRE_PEAK_VUS },
        { duration: '3m', target: PEAK_VUS },
        { duration: '2m', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.02'],
    'http_req_duration{name:POST /bulk-upload}': [`p(95)<${BULK_P95_MS}`, `p(99)<${BULK_P99_MS}`],
    login_failures: ['rate<0.02'],
    bulk_upload_failures: ['rate<0.03'],
    bulk_upload_duration: [`p(95)<${BULK_P95_MS}`, `p(99)<${BULK_P99_MS}`],
  },
};

function think(minSeconds, maxSeconds) {
  const t = minSeconds + Math.random() * (maxSeconds - minSeconds);
  sleep(t);
}

function parseSessionToken(setCookieHeader) {
  if (!setCookieHeader) {
    return '';
  }
  const match = setCookieHeader.match(/(?:^|[;,\s])franchise_session=([^;]+)/);
  return match && match[1] ? match[1] : '';
}

function authHeaders() {
  if (!sessionToken) {
    return {};
  }
  return {
    Cookie: `franchise_session=${sessionToken}`,
  };
}

function extractPendingToken(html) {
  const match = html.match(/name="pending_file_token"\s+value="([0-9a-f]{32})"/i);
  return match && match[1] ? match[1] : '';
}

function extractPendingFilename(html) {
  const match = html.match(/name="pending_original_filename"\s+value="([^"]*)"/i);
  return match && match[1] ? match[1] : 'bulk_upload.xlsx';
}

function extractUnknownEquipmentNames(html) {
  const names = [];
  const regex = /name="equip_([^"]+)"/g;
  let match;
  while ((match = regex.exec(html)) !== null) {
    if (match[1]) {
      names.push(match[1]);
    }
  }
  return Array.from(new Set(names));
}

function ensureLoggedIn() {
  if (loggedIn && sessionToken) {
    return true;
  }

  if (!LOGIN_ID || !LOGIN_PASSWORD) {
    return false;
  }

  const res = http.post(
    `${BASE_URL}/login`,
    {
      franchise_id: LOGIN_ID,
      password: LOGIN_PASSWORD,
    },
    {
      redirects: 0,
      tags: { name: 'POST /login' },
    }
  );

  const ok = check(res, {
    'login returns redirect': (r) => r.status === 302 || r.status === 303,
    'login sets session cookie': (r) => (r.headers['Set-Cookie'] || '').includes('franchise_session='),
  });

  sessionToken = parseSessionToken(res.headers['Set-Cookie']);
  loginFailureRate.add(!(ok && !!sessionToken));
  loggedIn = ok && !!sessionToken;
  return loggedIn;
}

function submitBulkUpload() {
  const fileName = BULK_FILE.split('/').pop().split('\\').pop();
  const bulkFile = http.file(
    bulkFixtureBytes,
    `${Date.now()}_${__VU}_${fileName}`,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  );

  const firstRes = http.post(
    `${BASE_URL}/bulk-upload`,
    { file: bulkFile },
    {
      headers: authHeaders(),
      tags: { name: 'POST /bulk-upload' },
      timeout: '180s',
    }
  );
  bulkUploadDuration.add(firstRes.timings.duration);

  const firstBody = (firstRes.body || '').toString();
  const needsClassification = firstBody.includes('Classify Your Equipment') && firstBody.includes('pending_file_token');

  let finalStatus = firstRes.status;
  let finalBody = firstBody;
  let finalOk = firstRes.status === 200 && (firstBody.includes('Upload Results') || needsClassification);

  if (needsClassification) {
    const pendingToken = extractPendingToken(firstBody);
    const pendingFilename = extractPendingFilename(firstBody);
    const unknownEquipment = extractUnknownEquipmentNames(firstBody);
    const classifyPayload = {
      pending_file_token: pendingToken,
      pending_original_filename: pendingFilename,
    };

    for (const equipName of unknownEquipment) {
      classifyPayload[`equip_${equipName}`] = 'Truck';
    }

    const secondRes = http.post(`${BASE_URL}/bulk-upload`, classifyPayload, {
      headers: authHeaders(),
      tags: { name: 'POST /bulk-upload' },
      timeout: '180s',
    });
    bulkUploadDuration.add(secondRes.timings.duration);
    finalStatus = secondRes.status;
    finalBody = (secondRes.body || '').toString();
    finalOk = secondRes.status === 200 && finalBody.includes('Upload Results');
  }

  const ok = check(
    { status: finalStatus, body: finalBody },
    {
      'bulk upload status 200': (r) => r.status === 200,
      'bulk upload results rendered': (r) => r.body.includes('Upload Results'),
    }
  );

  if (!(ok && finalOk)) {
    bulkHttpErrors.add(1);
    if (finalStatus >= 500) {
      bulkServerErrors.add(1);
    }
    if (bulkFailureSamples < MAX_BULK_FAILURE_SAMPLES) {
      console.error(`bulk upload failure sample status=${finalStatus} body=${finalBody.slice(0, 500)}`);
      bulkFailureSamples += 1;
    }
  }

  bulkFailureRate.add(!(ok && finalOk));
}

export default function () {
  group('bulk upload path', () => {
    const loginPage = http.get(`${BASE_URL}/login`, { tags: { name: 'GET /login' } });
    check(loginPage, {
      'guest login page 200': (r) => r.status === 200,
    });
    think(0.2, 0.6);

    if (!ensureLoggedIn()) {
      return;
    }

    submitBulkUpload();
    think(1.0, 2.0);
  });
}