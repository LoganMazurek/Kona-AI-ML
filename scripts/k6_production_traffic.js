import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

const BASE_URL = (__ENV.BASE_URL || 'https://loganmazurek.com').replace(/\/$/, '');
const LOGIN_ID = __ENV.LOGIN_ID || '';
const LOGIN_PASSWORD = __ENV.LOGIN_PASSWORD || '';
const ENABLE_PREDICT = (__ENV.ENABLE_PREDICT || 'false').toLowerCase() === 'true';
const PREDICT_SHARE = Number(__ENV.PREDICT_SHARE || '0.35');
const ENABLE_BULK_UPLOAD = (__ENV.ENABLE_BULK_UPLOAD || 'true').toLowerCase() === 'true';
const BULK_SHARE = Number(__ENV.BULK_SHARE || '1.0');
const PEAK_VUS = Number(__ENV.PEAK_VUS || '25');
const PRE_PEAK_VUS = Math.max(15, Math.floor(PEAK_VUS * 0.8));
const HTTP_P95_MS = Number(__ENV.HTTP_P95_MS || '1200');
const HTTP_P99_MS = Number(__ENV.HTTP_P99_MS || '2000');
const PREDICT_P95_MS = Number(__ENV.PREDICT_P95_MS || '2500');
const PREDICT_P99_MS = Number(__ENV.PREDICT_P99_MS || '4000');
const LOGIN_P95_MS = Number(__ENV.LOGIN_P95_MS || '800');
const LOGIN_P99_MS = Number(__ENV.LOGIN_P99_MS || '1500');
const DASHBOARD_P95_MS = Number(__ENV.DASHBOARD_P95_MS || '1200');
const DASHBOARD_P99_MS = Number(__ENV.DASHBOARD_P99_MS || '2500');
const BULK_P95_MS = Number(__ENV.BULK_P95_MS || '8000');
const BULK_P99_MS = Number(__ENV.BULK_P99_MS || '14000');

const INDUSTRIES = ['Festival', 'Corporate', 'Community', 'Education', 'Private'];
const EQUIPMENT = ['truck', 'blended truck', 'kiosk', 'mini'];
const CITIES = ['Naperville', 'Aurora', 'Plainfield', 'Joliet', 'Shorewood'];
const ZIPS = ['60540', '60504', '60564', '60435', '60431'];

const predictionDuration = new Trend('prediction_duration', true);
const bulkUploadDuration = new Trend('bulk_upload_duration', true);
const loginFailureRate = new Rate('login_failures');
const predictFailureRate = new Rate('predict_failures');
const bulkFailureRate = new Rate('bulk_upload_failures');
const predictHttpErrors = new Counter('predict_http_errors');
const predictAuthErrors = new Counter('predict_auth_errors');
const predictServerErrors = new Counter('predict_server_errors');
const bulkHttpErrors = new Counter('bulk_upload_http_errors');
const bulkServerErrors = new Counter('bulk_upload_server_errors');

let loggedIn = false;
let sessionToken = '';
let predictFailureSamples = 0;
let bulkFailureSamples = 0;
const MAX_PREDICT_FAILURE_SAMPLES = 5;
const MAX_BULK_FAILURE_SAMPLES = 5;

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

export const options = {
  scenarios: {
    production_mix: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '2m', target: 3 },
        { duration: '3m', target: 8 },
        { duration: '4m', target: 15 },
        { duration: '3m', target: PRE_PEAK_VUS },
        { duration: '2m', target: PEAK_VUS },
        { duration: '1m', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.02'],
    http_req_duration: [`p(95)<${HTTP_P95_MS}`, `p(99)<${HTTP_P99_MS}`],
    'http_req_duration{name:GET /login}': [`p(95)<${LOGIN_P95_MS}`, `p(99)<${LOGIN_P99_MS}`],
    'http_req_duration{name:GET / (guest)}': [`p(95)<${LOGIN_P95_MS}`, `p(99)<${LOGIN_P99_MS}`],
    'http_req_duration{name:GET /dashboard}': [`p(95)<${DASHBOARD_P95_MS}`, `p(99)<${DASHBOARD_P99_MS}`],
    'http_req_duration{name:GET / (auth)}': [`p(95)<${DASHBOARD_P95_MS}`, `p(99)<${DASHBOARD_P99_MS}`],
    'http_req_duration{name:POST /predict}': [`p(95)<${PREDICT_P95_MS}`, `p(99)<${PREDICT_P99_MS}`],
    'http_req_duration{name:GET /bulk-upload/template}': [`p(95)<${DASHBOARD_P95_MS}`, `p(99)<${DASHBOARD_P99_MS}`],
    'http_req_duration{name:POST /bulk-upload}': [`p(95)<${BULK_P95_MS}`, `p(99)<${BULK_P99_MS}`],
    login_failures: ['rate<0.02'],
    predict_failures: ['rate<0.03'],
    bulk_upload_failures: ['rate<0.03'],
    prediction_duration: [`p(95)<${PREDICT_P95_MS}`, `p(99)<${PREDICT_P99_MS}`],
    bulk_upload_duration: [`p(95)<${BULK_P95_MS}`, `p(99)<${BULK_P99_MS}`],
  },
};

function randomChoice(values) {
  return values[Math.floor(Math.random() * values.length)];
}

function randomFutureDate() {
  const daysAhead = 3 + Math.floor(Math.random() * 60);
  const d = new Date(Date.now() + daysAhead * 24 * 60 * 60 * 1000);
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function think(minSeconds, maxSeconds) {
  const t = minSeconds + Math.random() * (maxSeconds - minSeconds);
  sleep(t);
}

function extractPendingToken(html) {
  const match = html.match(/name="pending_file_token"\s+value="([0-9a-f]{32})"/i);
  return match && match[1] ? match[1] : '';
}

function extractPendingFilename(html) {
  const match = html.match(/name="pending_original_filename"\s+value="([^"]*)"/i);
  return match && match[1] ? match[1] : 'bulk_upload_template.xlsx';
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
      // Keep redirects disabled so auth failures do not look like successful logins.
      redirects: 0,
      tags: { name: 'POST /login' },
    }
  );

  const ok = check(res, {
    'login returns redirect': (r) => r.status === 302 || r.status === 303,
    'login sets session cookie': (r) => (r.headers['Set-Cookie'] || '').includes('franchise_session='),
  });

  sessionToken = parseSessionToken(res.headers['Set-Cookie']);
  loginFailureRate.add(!ok);
  loggedIn = ok && !!sessionToken;
  return ok;
}

function browseUnauthenticated() {
  group('guest browse', () => {
    const loginPage = http.get(`${BASE_URL}/login`, { tags: { name: 'GET /login' } });
    check(loginPage, {
      'guest login page 200': (r) => r.status === 200,
    });
    think(0.3, 1.0);

    // Unauthenticated root should redirect to login.
    const home = http.get(`${BASE_URL}/`, { redirects: 0, tags: { name: 'GET / (guest)' } });
    check(home, {
      'guest root redirect': (r) => r.status === 302 || r.status === 301,
    });
  });
}

function browseAuthenticated() {
  group('authenticated browse', () => {
    const dash = http.get(`${BASE_URL}/dashboard`, {
      redirects: 0,
      headers: authHeaders(),
      tags: { name: 'GET /dashboard' },
    });
    check(dash, {
      'dashboard authenticated 200': (r) => r.status === 200,
    });
    think(0.5, 1.5);

    const home = http.get(`${BASE_URL}/`, {
      redirects: 0,
      headers: authHeaders(),
      tags: { name: 'GET / (auth)' },
    });
    check(home, {
      'auth home 200': (r) => r.status === 200,
    });

    if (dash.status === 302 || dash.status === 301 || home.status === 302 || home.status === 301) {
      loggedIn = false;
      sessionToken = '';
    }
  });
}

function submitPrediction() {
  if (!ENABLE_PREDICT) {
    return;
  }

  const payload = {
    event_name: `Load Test Event ${Date.now()}`,
    industry: randomChoice(INDUSTRIES),
    equipment_type: randomChoice(EQUIPMENT),
    event_date: randomFutureDate(),
    start_time: '11:00',
    duration: (2 + Math.random() * 4).toFixed(1),
    city: randomChoice(CITIES),
    zip_code: randomChoice(ZIPS),
    temperature: String(45 + Math.floor(Math.random() * 40)),
    is_test: '1',
    is_booked: Math.random() < 0.4 ? '1' : '0',
  };

  const res = http.post(`${BASE_URL}/predict`, payload, {
    headers: authHeaders(),
    tags: { name: 'POST /predict' },
    timeout: '45s',
  });

  predictionDuration.add(res.timings.duration);

  const ok = check(res, {
    'predict status 200': (r) => r.status === 200,
    'predict returns json': (r) => (r.headers['Content-Type'] || '').includes('application/json'),
    'predict success true': (r) => {
      try {
        return r.json('success') === true;
      } catch (e) {
        return false;
      }
    },
  });

  if (!ok) {
    predictHttpErrors.add(1);
    if (res.status === 401 || res.status === 403) {
      predictAuthErrors.add(1);
      loggedIn = false;
      sessionToken = '';
    }
    if (res.status >= 500) {
      predictServerErrors.add(1);
    }

    if (predictFailureSamples < MAX_PREDICT_FAILURE_SAMPLES) {
      const body = (res.body || '').toString().slice(0, 500);
      console.error(`predict failure sample status=${res.status} body=${body}`);
      predictFailureSamples += 1;
    }
  }

  predictFailureRate.add(!ok);
}

function submitBulkUpload() {
  if (!ENABLE_BULK_UPLOAD) {
    return;
  }

  const templateRes = http.get(`${BASE_URL}/bulk-upload/template`, {
    headers: authHeaders(),
    tags: { name: 'GET /bulk-upload/template' },
    timeout: '45s',
  });

  const templateOk = check(templateRes, {
    'bulk template status 200': (r) => r.status === 200,
  });

  if (!templateOk) {
    bulkHttpErrors.add(1);
    bulkFailureRate.add(true);
    return;
  }

  const bulkFile = http.file(
    templateRes.body,
    `bulk_upload_${Date.now()}_${__VU}.xlsx`,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  );

  const firstRes = http.post(
    `${BASE_URL}/bulk-upload`,
    { file: bulkFile },
    {
      headers: authHeaders(),
      tags: { name: 'POST /bulk-upload' },
      timeout: '120s',
    }
  );
  bulkUploadDuration.add(firstRes.timings.duration);

  const firstBody = (firstRes.body || '').toString();
  const needsClassification = firstBody.includes('Classify Your Equipment') && firstBody.includes('pending_file_token');
  const firstPass = firstRes.status === 200 && (firstBody.includes('Upload Results') || needsClassification);

  let finalOk = firstPass;
  let finalStatus = firstRes.status;
  let finalBody = firstBody;

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
      timeout: '120s',
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

  if (!ok || !finalOk) {
    bulkHttpErrors.add(1);
    if (finalStatus >= 500) {
      bulkServerErrors.add(1);
    }

    if (bulkFailureSamples < MAX_BULK_FAILURE_SAMPLES) {
      const body = finalBody.slice(0, 500);
      console.error(`bulk upload failure sample status=${finalStatus} body=${body}`);
      bulkFailureSamples += 1;
    }
  }

  bulkFailureRate.add(!(ok && finalOk));
}

export default function () {
  browseUnauthenticated();
  think(0.2, 0.8);

  if (!ensureLoggedIn()) {
    return;
  }

  browseAuthenticated();
  think(0.6, 1.8);

  if (ENABLE_BULK_UPLOAD && Math.random() < BULK_SHARE) {
    group('bulk upload path', () => {
      submitBulkUpload();
      think(1.0, 2.5);
    });
  }

  if (ENABLE_PREDICT && Math.random() < PREDICT_SHARE) {
    group('prediction path', () => {
      submitPrediction();
      think(0.5, 1.5);
    });
  }
}
