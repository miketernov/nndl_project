// Telco Customer Churn — simplified data loading with PapaParse
//---------------------------------------------------------------

// global vars
let rawTrain = null;
let rawTest = null;

const byId = (id) => document.getElementById(id);
const info = (msg) => (byId('data-status').innerHTML = msg);

// read file as text
function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

// normalize telco row
function normalizeTelcoRow(o) {
  if (!o) return {};
  const r = { ...o };
  for (let k in r) {
    if (typeof r[k] === 'string') r[k] = r[k].trim();
  }

  // clean TotalCharges
  r.TotalCharges = r.TotalCharges === '' ? null : parseFloat(r.TotalCharges);
  r.tenure = parseFloat(r.tenure);
  r.MonthlyCharges = parseFloat(r.MonthlyCharges);

  // convert Yes/No
  const yesNoCols = [
    'Partner','Dependents','PhoneService','PaperlessBilling',
    'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies'
  ];
  yesNoCols.forEach(c => {
    if (r[c] != null) {
      const val = String(r[c]).toLowerCase();
      r[c] = val.includes('yes') ? 'Yes' : 'No';
    }
  });

  // SeniorCitizen: 1 → Yes
  if (r.SeniorCitizen != null)
    r.SeniorCitizen = String(r.SeniorCitizen) === '1' ? 'Yes' : 'No';

  // target Churn: Yes/No → 1/0
  if (r.Churn != null)
    r.Churn = r.Churn.trim().toLowerCase() === 'yes' ? 1 : 0;

  return r;
}

// create preview HTML table
function headTable(rows, limit = 10) {
  if (!rows || !rows.length) return '<p>No data</p>';
  const cols = Object.keys(rows[0]);
  let html = '<div style="overflow:auto;max-height:380px"><table><thead><tr>';
  html += cols.map(c => `<th>${c}</th>`).join('');
  html += '</tr></thead><tbody>';
  rows.slice(0, limit).forEach(r => {
    html += '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

// Load CSVs
async function loadData() {
  const trainFile = byId('train-file').files[0];
  const testFile = byId('test-file').files[0];
  if (!trainFile || !testFile) {
    alert('Please upload BOTH train.csv and test.csv files.');
    return;
  }

  info('Loading CSV files…');

  try {
    // TRAIN
    const trainText = await readFile(trainFile);
    const trainParsed = Papa.parse(trainText, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true
    });
    rawTrain = trainParsed.data.map(normalizeTelcoRow);

    // TEST
    const testText = await readFile(testFile);
    const testParsed = Papa.parse(testText, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true
    });
    rawTest = testParsed.data.map(normalizeTelcoRow);

    // summary
    const nTrain = rawTrain.length;
    const nCols = Object.keys(rawTrain[0] || {}).length;
    const posCnt = rawTrain.filter(r => r.Churn === 1).length;
    const churnRate = (posCnt / nTrain * 100).toFixed(2);

    info(`✅ Loaded: Train ${nTrain}×${nCols}, Test ${rawTest.length} rows`);
    byId('data-preview').innerHTML = headTable(rawTrain, 10);

    console.log('Train sample:', rawTrain[0]);
    console.log('Test sample:', rawTest[0]);
    console.log(`Churn rate: ${churnRate}%`);
  } catch (err) {
    console.error(err);
    info(`❌ Error loading data: ${err.message}`);
  }
}

document.getElementById('load-data-btn').addEventListener('click', loadData);
