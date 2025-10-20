/*************************************************
 * Telco Customer Churn — full in-browser pipeline
 * EDA + preprocessing + model + training + metrics
 *************************************************/

// --------- Globals ---------
let rawTrain = null, rawTest = null;
const byId = id => document.getElementById(id);
const info = msg => byId('data-status').innerHTML = msg;

// ===== Feature schema =====
const NUM_COLS = ['tenure','MonthlyCharges','TotalCharges'];
const BIN_COLS = [
  'Partner','Dependents','PhoneService','PaperlessBilling',
  'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
  'TechSupport','StreamingTV','StreamingMovies','SeniorCitizen'
];
const OH_COLS = ['Contract','InternetService','PaymentMethod','gender']; // one-hot
const TARGET = 'Churn';
const IDCOL  = 'customerID';

// Fit-time artifacts
const prep = {
  numMedian:{}, numMean:{}, numStd:{},
  ohLevels:{},
  trainX:null, trainY:null, valX:null, valY:null,
  testX:null, testIDs:[]
};

let model = null;
let valPreds = null;
let rocChart = null;

// --------- Utils ---------
function readFile(file){
  return new Promise((res,rej)=>{
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.onerror = () => rej(new Error('Failed to read file'));
    r.readAsText(file);
  });
}

function normalizeRow(o){
  // trim everything
  const r = {...o};
  Object.keys(r).forEach(k => { if (typeof r[k] === 'string') r[k] = r[k].trim(); });

  // Churn: Yes/No -> 1/0 (train only)
  if (r.Churn !== undefined) r.Churn = String(r.Churn).toLowerCase()==='yes'?1:0;

  // SeniorCitizen: 1/0 -> Yes/No, чтобы быть в BIN_COLS
  if (r.SeniorCitizen !== undefined)
    r.SeniorCitizen = String(r.SeniorCitizen)==='1'?'Yes':'No';

  // numeric columns to number (may be null here — we impute later)
  ['tenure','MonthlyCharges','TotalCharges'].forEach(c=>{
    const n = Number(r[c]); r[c] = Number.isFinite(n) ? n : null;
  });

  return r;
}


// Стековая диаграмма: доли Churn Yes/No по категориям колонки (Contract, InternetService)
function buildStackedChurnChart(rows, colName, ctx){
  const groups = {}; // {level:{yes:count, no:count}}
  rows.forEach(r=>{
    const level = (r[colName]==null || r[colName]==='') ? 'NA' : String(r[colName]);
    if (!groups[level]) groups[level] = {yes:0, no:0};
    if (r.Churn===1) groups[level].yes++; else groups[level].no++;
  });
  const labels = Object.keys(groups);
  const yesPct = labels.map(l=>{
    const g = groups[l]; const tot = g.yes+g.no || 1;
    return Math.round((g.yes/tot)*1000)/10; // %
  });
  const noPct = labels.map(l=>{
    const g = groups[l]; const tot = g.yes+g.no || 1;
    return Math.round((g.no/tot)*1000)/10;
  });

  new Chart(ctx,{
    type:'bar',
    data:{
      labels,
      datasets:[
        {label:'No',  data:noPct, backgroundColor:'#22c55e', stack:'churn'},
        {label:'Yes', data:yesPct, backgroundColor:'#ef4444', stack:'churn'}
      ]
    },
    options:{
      maintainAspectRatio:false,
      layout:{padding:{bottom:10,top:10}},
      plugins:{
        legend:{position:'top', labels:{color:'#e5e7eb', boxWidth:14}},
        title:{display:false}
      },
      scales:{
        x:{
          stacked:true,
          ticks:{color:'#e5e7eb', maxRotation:0, minRotation:0},
          grid:{display:false}
        },
        y:{
          stacked:true,
          beginAtZero:true,
          ticks:{color:'#e5e7eb', callback:v=>v+'%'},
          grid:{color:'rgba(148,163,184,0.15)'}
        }
      }
    }
  });
}


// Две гистограммы (Yes/No) поверх друг друга для числового признака
function buildDualHistogram(rows, colName, ctx, bins = 20) {
  const valsNo  = rows.filter(r => r.Churn === 0).map(r => Number(r[colName])).filter(Number.isFinite);
  const valsYes = rows.filter(r => r.Churn === 1).map(r => Number(r[colName])).filter(Number.isFinite);
  if (!valsNo.length && !valsYes.length) return;

  const minV = Math.min(...valsNo, ...valsYes);
  const maxV = Math.max(...valsNo, ...valsYes);
  const step = (maxV - minV) / bins || 1;

  const hist = (values) => {
    const counts = new Array(bins).fill(0);
    values.forEach(v => {
      let idx = Math.floor((v - minV) / step);
      if (idx < 0) idx = 0;
      if (idx >= bins) idx = bins - 1;
      counts[idx]++;
    });
    return counts;
  };

  const hNo  = hist(valsNo);
  const hYes = hist(valsYes);

  const labels = Array.from({ length: bins }, (_, i) => {
    const a = minV + i * step, b = a + step;
    return `${a.toFixed(0)}–${b.toFixed(0)}`;
  });

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'No',  data: hNo,  backgroundColor: 'rgba(34,197,94,0.55)' },
        { label: 'Yes', data: hYes, backgroundColor: 'rgba(239,68,68,0.55)' }
      ]
    },
    options: {
      maintainAspectRatio: false,
      layout: { padding: { bottom: 10, top: 10 } },
      plugins: {
        legend: { position: 'top', labels: { color: '#e5e7eb', boxWidth: 14 } },
        title: { display: false }
      },
      scales: {
        x: {
          ticks: {
            color: '#e5e7eb',
            autoSkip: true,
            maxRotation: 45,
            minRotation: 45,
            // показываем подпись только на каждом втором тике
            callback: (value, index) => (index % 2 === 0 ? labels[index] : '')
          },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#e5e7eb' },
          grid: { color: 'rgba(148,163,184,0.15)' }
        }
      }
    }
  });
}

function headTable(rows, limit=10){
  if (!rows || !rows.length) return '<p>No data</p>';
  const cols = Object.keys(rows[0]);
  let html = '<div class="scroll"><table><thead><tr>';
  html += cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  rows.slice(0,limit).forEach(r=>{
    html += '<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

function toNum(v){ const n = Number(v); return Number.isFinite(n)? n : null; }
function ynTo01(v){
  const s = String(v).toLowerCase();
  if (['yes','y','true','1','male'].includes(s)) return 1;
  if (['no','n','false','0','female'].includes(s)) return 0;
  return null; // позже заменим на 0
}

// --------- Load Data ---------
async function loadData(){
  const trainFile = byId('train-file').files[0];
  const testFile  = byId('test-file').files[0];
  if (!trainFile || !testFile){ alert('Upload BOTH train.csv and test.csv'); return; }

  info('Loading CSV files…');
  try{
    const trainParsed = Papa.parse(await readFile(trainFile), {header:true,skipEmptyLines:true});
    const testParsed  = Papa.parse(await readFile(testFile),  {header:true,skipEmptyLines:true});
    rawTrain = trainParsed.data.map(normalizeRow);
    rawTest  = testParsed.data.map(normalizeRow);

    const n = rawTrain.length;
    const churnRate = (100*rawTrain.filter(r=>r.Churn===1).length/n).toFixed(2);
    info(`✅ Loaded: Train ${n}, Test ${rawTest.length} | Churn ${churnRate}%`);

    byId('data-preview').innerHTML = headTable(rawTrain,10);
    byId('eda-btn').disabled = false;
    byId('create-model-btn').disabled = false;
  }catch(err){ console.error(err); info(`❌ ${err.message}`); }
}

// --------- EDA ---------
function computeCorrelation(rows, cols){
  const corr = {};
  cols.forEach(a=>{
    corr[a] = {};
    cols.forEach(b=>{
      const pairs = rows
        .map(r => [parseFloat(r[a]), parseFloat(r[b])])
        .filter(([x,y]) => !isNaN(x) && !isNaN(y));
      if (!pairs.length){ corr[a][b]=0; return; }
      const xs = pairs.map(p=>p[0]), ys = pairs.map(p=>p[1]);
      const mx = xs.reduce((s,v)=>s+v,0)/xs.length;
      const my = ys.reduce((s,v)=>s+v,0)/ys.length;
      const num = xs.map((v,i)=>(v-mx)*(ys[i]-my)).reduce((a,b)=>a+b,0);
      const den = Math.sqrt(
        xs.map(v=>(v-mx)**2).reduce((a,b)=>a+b,0)*
        ys.map(v=>(v-my)**2).reduce((a,b)=>a+b,0)
      );
      corr[a][b] = den ? num/den : 0;
    });
  });
  return corr;
}

function runEDA(){
  try{
    if (!rawTrain || !rawTrain.length) { alert('Load train data first'); return; }

    const container = byId('data-preview');
    container.innerHTML = "<h3>Data Preview</h3>" + headTable(rawTrain,10);

    // safe columns
    const sample = rawTrain.find(r => r && Object.keys(r).length) || {};
    const columns = Object.keys(sample);

    // Data types
    const dataTypes = columns.map(col=>{
      const vals = rawTrain.map(r=> r?.[col]).filter(v=> v!==undefined);
      const nonNull = vals.find(v => v!==null && v!=='');
      let dtype = typeof nonNull;
      if (nonNull!==undefined && nonNull!==null && nonNull!=='' && !isNaN(parseFloat(nonNull))) dtype='number';
      else if (['yes','no','male','female'].includes(String(nonNull||'').toLowerCase())) dtype='category';
      else if (String(nonNull||'').length>30) dtype='text';
      return {col, dtype};
    });
    let dtypeHTML = "<h3>Data Types Overview</h3><div class='scroll'><table><tr><th>Feature</th><th>Detected Type</th></tr>";
    dataTypes.forEach(d=> dtypeHTML += `<tr><td>${d.col}</td><td>${d.dtype}</td></tr>`);
    dtypeHTML += "</table></div>";

    // numeric cols (exclude ID)
    let numericCols = columns.filter(key=>{
      if (key.toLowerCase()==='customerid') return false;
      return rawTrain.some(r=>{
        const v = r?.[key];
        return v!=='' && v!==null && !isNaN(parseFloat(v));
      });
    });

    // Top-5 binary categories correlated with Churn
    const hasChurn = rawTrain.some(r=> r?.Churn !== undefined);
    if (hasChurn){
      const catCols = columns.filter(c => c.toLowerCase()!=='churn' && !numericCols.includes(c));
      const catCorr=[];
      catCols.forEach(col=>{
        const vals = rawTrain.map(r=> r?.[col]);
        const uniq = [...new Set(vals.map(v=> String(v??'').toLowerCase()))].filter(u=>u!=='');
        if (uniq.length===2){
          const map={}; map[uniq[0]]=0; map[uniq[1]]=1;
          const x = rawTrain.map(r=> map[String(r?.[col]??'').toLowerCase()]);
          const y = rawTrain.map(r=> Number(r?.Churn));
          const valid = x.map((v,i)=>[v,y[i]]).filter(([a,b])=> a!==undefined && !isNaN(a) && !isNaN(b));
          if (valid.length>5){
            const xs = valid.map(p=>p[0]), ys = valid.map(p=>p[1]);
            const mx = xs.reduce((a,b)=>a+b,0)/xs.length;
            const my = ys.reduce((a,b)=>a+b,0)/ys.length;
            const num = xs.map((v,i)=>(v-mx)*(ys[i]-my)).reduce((a,b)=>a+b,0);
            const den = Math.sqrt(
              xs.map(v=>(v-mx)**2).reduce((a,b)=>a+b,0) *
              ys.map(v=>(v-my)**2).reduce((a,b)=>a+b,0)
            );
            const c = den? num/den : 0;
            catCorr.push({col, corr:c});
          }
        }
      });
      catCorr.sort((a,b)=> Math.abs(b.corr)-Math.abs(a.corr));
      const topCats = catCorr.slice(0,5).map(o=>o.col);
      numericCols = [...numericCols, ...topCats];
    }

    // temporary map Yes/No→0/1 for correlation
    const processed = rawTrain.map(row=>{
      const r = {...row};
      Object.keys(r).forEach(k=>{
        const v = String(r[k]).toLowerCase();
        if (['yes','male','true','1'].includes(v)) r[k]=1;
        else if (['no','female','false','0'].includes(v)) r[k]=0;
      });
      return r;
    });

    const corrM = computeCorrelation(processed, numericCols);
    let corrHTML = "<h3>Correlation Matrix</h3><div class='scroll'><table><tr><th></th>";
    numericCols.forEach(c=> corrHTML += `<th>${c}</th>`);
    corrHTML += "</tr>";
    numericCols.forEach(a=>{
      corrHTML += `<tr><th>${a}</th>`;
      numericCols.forEach(b=>{
        let v = corrM[a][b]; if (isNaN(v)) v=0;
        const color = v>=0? `rgba(56,189,248,${Math.abs(v)})` : `rgba(239,68,68,${Math.abs(v)})`;
        corrHTML += `<td style="background:${color};color:#fff;text-align:center;">${v.toFixed(2)}</td>`;
      });
      corrHTML += "</tr>";
    });
    corrHTML += "</table></div>";

    // missing values
    let missHTML = "<h3>Missing Values</h3><div class='scroll'><table><tr><th>Feature</th><th>Missing %</th></tr>";
    columns.forEach(c=>{
      const miss = rawTrain.filter(r => r?.[c]===null || r?.[c]==='').length;
      const pct = (miss/rawTrain.length*100).toFixed(1);
      missHTML += `<tr><td>${c}</td><td>${pct}%</td></tr>`;
    });
    missHTML += "</table></div>";

    // numeric summary
    let numHTML = "<h3>Numeric Summary</h3><div class='scroll'><table><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>";
    const numForSummary = numericCols.filter(c=>c.toLowerCase()!=='customerid');
    numForSummary.forEach(c=>{
      const vals = rawTrain.map(r=> parseFloat(r?.[c])).filter(v=> !isNaN(v));
      if (!vals.length) return;
      const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
      const std  = Math.sqrt(vals.map(v=>(v-mean)**2).reduce((a,b)=>a+b,0)/vals.length);
      const min = Math.min(...vals), max = Math.max(...vals);
      numHTML += `<tr><td>${c}</td><td>${mean.toFixed(2)}</td><td>${std.toFixed(2)}</td><td>${min.toFixed(2)}</td><td>${max.toFixed(2)}</td></tr>`;
    });
    numHTML += "</table></div>";

    // churn distribution (общая)
    const yes = rawTrain.filter(r=>r.Churn===1).length;
    const no  = rawTrain.length - yes;
    const churnChartBox = `
      <h3>Churn Distribution</h3>
      <div class="chartbox"><canvas id="churnChart"></canvas></div>
    `;

    // +++ НОВЫЕ ГРАФИКИ +++
    const extraChartsHTML = `
      <div class="grid2 mt20">
        <div class="chartbox">
          <h3>Churn by Contract (stacked)</h3>
          <canvas id="contractChurnChart"></canvas>
        </div>
        <div class="chartbox">
          <h3>Churn by Internet Service (stacked)</h3>
          <canvas id="internetChurnChart"></canvas>
        </div>
      </div>
      <div class="grid2 mt20">
        <div class="chartbox">
          <h3>Monthly Charges — distribution (Yes/No)</h3>
          <canvas id="chargesHistChart"></canvas>
        </div>
        <div class="chartbox">
          <h3>Tenure — distribution (Yes/No)</h3>
          <canvas id="tenureHistChart"></canvas>
        </div>
      </div>
    `;

    // layout
    container.innerHTML += `
      <div class="grid2 mt20">
        <div>${dtypeHTML}</div>
        <div>${corrHTML}</div>
      </div>
      <div class="grid2 mt20">
        <div>${missHTML}</div>
        <div>${numHTML}</div>
      </div>
      <div class="mt20">${churnChartBox}</div>
      ${extraChartsHTML}
    `;

    // --- Charts render ---
    // 1) общий churn
    new Chart(byId('churnChart').getContext('2d'),{
      type:'bar',
      data:{labels:['No','Yes'], datasets:[{label:'Churn Count', data:[no,yes], backgroundColor:['#22c55e','#ef4444']}]},
      options:{maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}
    });

    // 2) stacked: Contract
    buildStackedChurnChart(
      rawTrain,
      'Contract',
      byId('contractChurnChart').getContext('2d')
    );

    // 3) stacked: InternetService
    buildStackedChurnChart(
      rawTrain,
      'InternetService',
      byId('internetChurnChart').getContext('2d')
    );

    // 4) hist: MonthlyCharges (Yes/No)
    buildDualHistogram(
      rawTrain,
      'MonthlyCharges',
      byId('chargesHistChart').getContext('2d'),
      20
    );

    // 5) hist: Tenure (Yes/No)
    buildDualHistogram(
      rawTrain,
      'tenure',
      byId('tenureHistChart').getContext('2d'),
      20
    );

    info('✅ EDA complete');
  }catch(e){
    console.error(e);
    alert('EDA error: ' + e.message);
  }
}

// --------- Preprocessing for model (NO row drops; we impute) ---------
function fitPreprocess(trainRows){
  // numeric stats
  NUM_COLS.forEach(c=>{
    const vals = trainRows.map(r=> toNum(r[c])).filter(v=> v!=null);
    const sorted = [...vals].sort((a,b)=>a-b);
    const mid = Math.floor(sorted.length/2);
    const median = sorted.length? (sorted.length%2? sorted[mid] : (sorted[mid-1]+sorted[mid])/2) : 0;
    const mean = vals.length? vals.reduce((s,x)=>s+x,0)/vals.length : 0;
    const std  = Math.sqrt(vals.reduce((s,x)=> s+(x-mean)**2,0)/(vals.length||1)) || 1;
    prep.numMedian[c]=median; prep.numMean[c]=mean; prep.numStd[c]=std;
  });
  // one-hot levels from train
  OH_COLS.forEach(c=>{
    const levels = Array.from(new Set(trainRows.map(r=> (r[c]==null?'NA':String(r[c])) )));
    prep.ohLevels[c]=levels;
  });
}

function transformRow(r){
  // numbers: impute median + z-score
  const nums = NUM_COLS.map(c=>{
    let v = toNum(r[c]); if (v==null) v = prep.numMedian[c];
    return (v - prep.numMean[c]) / prep.numStd[c];
  });
  // binaries: Yes/No -> 1/0 (null → 0)
  const bins = BIN_COLS.map(c=>{
    const v = ynTo01(r[c]); return v==null ? 0 : v;
  });
  // one-hot with fixed train levels (+ NA)
  const oh = OH_COLS.flatMap(c=>{
    const levels = prep.ohLevels[c];
    const v = r[c]==null ? 'NA' : String(r[c]);
    return levels.map(l => (l===v?1:0));
  });
  return nums.concat(bins, oh);
}

function makeTensors(trainRows, testRows){
  fitPreprocess(trainRows);

  const X = trainRows.map(transformRow);
  const y = trainRows.map(r=> Number(r[TARGET]));

  // split 80/20
  const cut = Math.floor(X.length*0.8);
  prep.trainX = tf.tensor2d(X.slice(0,cut));
  prep.trainY = tf.tensor1d(y.slice(0,cut));
  prep.valX   = tf.tensor2d(X.slice(cut));
  prep.valY   = tf.tensor1d(y.slice(cut));

  prep.testIDs= testRows.map(r=> r[IDCOL]);
  prep.testX  = tf.tensor2d(testRows.map(transformRow));

  return prep.trainX.shape[1];
}

// --------- Model ---------
function buildModel(inputDim){
  const m = tf.sequential();
  m.add(tf.layers.dense({units:128, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.dropout({rate:0.3}));
  m.add(tf.layers.dense({units:64, activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.2}));
  m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  m.compile({ optimizer: tf.train.adam(0.001), loss:'binaryCrossentropy', metrics:['accuracy'] });
  return m;
}

function computeClassWeights(y){
  const n=y.length, pos=y.reduce((s,v)=>s+(v===1?1:0),0), neg=n-pos;
  return {0: n/(2*(neg||1)), 1: n/(2*(pos||1))};
}

// --------- Create & Train ---------
function onCreateModel(){
  if(!rawTrain||!rawTest){ alert('Load data first'); return; }
  const inputDim = makeTensors(rawTrain, rawTest);  // <- fit + tensors
  model = buildModel(inputDim);
  byId('train-btn').disabled = false;
  byId('training-status').innerText =
    `Model ready. Input dim: ${inputDim}. Params: ${model.countParams()}`;
}

async function onTrain(){
  if(!model){ alert('Create model first'); return; }
  const status = byId('training-status');

  // charts under block
  const lossChart = new Chart(byId('lossChart').getContext('2d'),{
    type:'line', data:{labels:[], datasets:[{label:'Loss', data:[], borderColor:'#ef4444', tension:0.2}]},
    options:{maintainAspectRatio:false, scales:{y:{beginAtZero:true}}}
  });
  const accChart = new Chart(byId('accuracyChart').getContext('2d'),{
    type:'line', data:{labels:[], datasets:[{label:'Accuracy', data:[], borderColor:'#22c55e', tension:0.2}]},
    options:{maintainAspectRatio:false, scales:{y:{beginAtZero:true, max:1}}}
  });

  const E=40, B=64, cw = computeClassWeights(prep.trainY.arraySync());
  let best=Infinity, patience=6, wait=0;

  await model.fit(prep.trainX, prep.trainY, {
    epochs:E, batchSize:B, validationData:[prep.valX, prep.valY], classWeight:cw,
    callbacks:{
      onEpochEnd: async (ep, logs)=>{
        lossChart.data.labels.push(ep+1);
        lossChart.data.datasets[0].data.push(logs.loss);
        lossChart.update();
        accChart.data.labels.push(ep+1);
        accChart.data.datasets[0].data.push(logs.acc);
        accChart.update();

        status.innerText =
          `Epoch ${ep+1}/${E} | loss ${logs.loss.toFixed(4)} acc ${logs.acc.toFixed(3)} | vloss ${logs.val_loss.toFixed(4)} vacc ${logs.val_acc.toFixed(3)}`;

        if (logs.val_loss < best-1e-4){ best=logs.val_loss; wait=0; }
        else if (++wait>=patience){ status.innerText += ' | Early stopped'; this.modelStop=true; }
        await tf.nextFrame();
      }
    }
  });

  valPreds = model.predict(prep.valX);
  byId('threshold-slider').disabled = false;
  byId('threshold-slider').value = 0.35;
  byId('threshold-value').textContent = '0.35';
  updateMetrics();

  byId('predict-btn').disabled = false;
}

// --------- Metrics & ROC ---------
function updateMetrics(){
  if (!valPreds) return;
  const thr = parseFloat(byId('threshold-slider').value);
  byId('threshold-value').textContent = thr.toFixed(2);

  const yPred = valPreds.arraySync().map(v=> Array.isArray(v)? v[0]:v);
  const yTrue = prep.valY.arraySync();

  let tp=0,tn=0,fp=0,fn=0;
  for (let i=0;i<yPred.length;i++){
    const p = yPred[i]>=thr?1:0, a = yTrue[i];
    if (p===1&&a===1) tp++; else if (p===0&&a===0) tn++;
    else if (p===1&&a===0) fp++; else fn++;
  }
  const precision = tp/(tp+fp)||0, recall = tp/(tp+fn)||0;
  const f1 = 2*(precision*recall)/(precision+recall)||0;
  const acc = (tp+tn)/(tp+tn+fp+fn)||0;

  byId('confusion-matrix').innerHTML =
    `<h3>Confusion Matrix</h3>
     <table>
       <tr><th></th><th>Pred +</th><th>Pred -</th></tr>
       <tr><th>Actual +</th><td>${tp}</td><td>${fn}</td></tr>
       <tr><th>Actual -</th><td>${fp}</td><td>${tn}</td></tr>
     </table>`;

  byId('performance-metrics').innerHTML =
    `<h3>Metrics</h3>
     <p>Accuracy: ${(acc*100).toFixed(2)}%</p>
     <p>Precision: ${precision.toFixed(3)}</p>
     <p>Recall: ${recall.toFixed(3)}</p>
     <p>F1: ${f1.toFixed(3)}</p>`;

  // ROC + AUC
  const steps=100, pts=[];
  for(let t=0;t<=1;t+=1/steps){
    let TP=0,FP=0,FN=0,TN=0;
    for(let i=0;i<yPred.length;i++){
      const p=yPred[i]>=t?1:0, a=yTrue[i];
      if (a===1&&p===1) TP++; if (a===0&&p===1) FP++;
      if (a===1&&p===0) FN++; if (a===0&&p===0) TN++;
    }
    const TPR = TP/(TP+FN)||0, FPR = FP/(FP+TN)||0;
    pts.push({x:FPR, y:TPR});
  }
  let auc=0; for(let i=1;i<pts.length;i++){ const dx=pts[i].x-pts[i-1].x; const h=(pts[i].y+pts[i-1].y)/2; auc += dx*h; }
  const ctx = byId('rocChart').getContext('2d');
  if (rocChart) rocChart.destroy();
  rocChart = new Chart(ctx,{
    type:'line',
    data:{labels:pts.map(p=>p.x.toFixed(2)), datasets:[{label:`ROC (AUC=${auc.toFixed(3)})`, data:pts.map(p=>p.y), fill:false}]},
    options:{maintainAspectRatio:false, plugins:{legend:{display:true}},
      scales:{x:{title:{display:true,text:'FPR'}}, y:{title:{display:true,text:'TPR'},min:0,max:1}}}
  });
}
byId('threshold-slider').addEventListener('input', updateMetrics);

// --------- Predict & Export ---------
function predictOnTest(){
  if(!model||!prep.testX){ alert('Train the model first'); return; }
  const thr = parseFloat(byId('threshold-slider').value||'0.5');
  const probs = model.predict(prep.testX).arraySync().map(v=> Array.isArray(v)? v[0]:v);
  const rows = probs.map((p,i)=>({customerID:prep.testIDs[i], prediction:(p>=thr?1:0), probability:p}));
  const cols=['customerID','prediction','probability'];
  let html = '<div class="scroll"><table><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  rows.slice(0,20).forEach(r=>{
    html += '<tr>'+cols.map(c=>`<td>${c==='probability'? r[c].toFixed(4): r[c]}</td>`).join('')+'</tr>';
  });
  html += '</tbody></table></div>';
  byId('prediction-output').innerHTML = html;
  byId('export-btn').disabled = false;
}

function exportCSV(){
  if(!model||!prep.testX){ alert('Nothing to export'); return; }
  const thr = parseFloat(byId('threshold-slider').value||'0.5');
  const probs = model.predict(prep.testX).arraySync().map(v=> Array.isArray(v)? v[0]:v);
  let csv='customerID,prediction,probability\n';
  probs.forEach((p,i)=>{ csv+=`${prep.testIDs[i]},${p>=thr?1:0},${p.toFixed(6)}\n`; });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='telco_churn_predictions.csv'; a.click();
}

// --------- Events (EDA guarded) ---------
byId('load-data-btn').addEventListener('click', loadData);
byId('eda-btn').addEventListener('click', () => { try { runEDA(); } catch(e){ console.error(e); alert('EDA error: '+e.message);} });
byId('create-model-btn').addEventListener('click', onCreateModel);
byId('train-btn').addEventListener('click', onTrain);
byId('predict-btn').addEventListener('click', predictOnTest);
byId('export-btn').addEventListener('click', exportCSV);
