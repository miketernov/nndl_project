//---------------------------------------------------------------
// Telco Customer Churn - Full TensorFlow.js App
//---------------------------------------------------------------

let rawTrain = null, rawTest = null;
let model = null, valData = null, valLabels = null;
let validationPreds = null, testPreds = null;
const byId = id => document.getElementById(id);
const info = msg => byId('data-status').innerHTML = msg;

// === FILE READING ===
function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

// === DATA CLEANING ===
function normalizeTelcoRow(o) {
  if (!o) return {};
  const r = { ...o };
  for (let k in r) if (typeof r[k] === 'string') r[k] = r[k].trim();
  r.TotalCharges = r.TotalCharges === '' ? null : parseFloat(r.TotalCharges);
  r.tenure = parseFloat(r.tenure);
  r.MonthlyCharges = parseFloat(r.MonthlyCharges);
  const yesNo = ['Partner','Dependents','PhoneService','PaperlessBilling',
    'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies'];
  yesNo.forEach(c=>{
    if (r[c]) {
      const val = r[c].toLowerCase();
      r[c] = val.includes('yes') ? 'Yes':'No';
    }
  });
  if (r.SeniorCitizen!=null)
    r.SeniorCitizen = String(r.SeniorCitizen)==='1'?'Yes':'No';
  if (r.Churn!=null)
    r.Churn = r.Churn.toLowerCase()==='yes'?1:0;
  return r;
}

// === PREVIEW TABLE ===
function headTable(rows, limit=10){
  if (!rows || !rows.length) return '<p>No data</p>';
  const cols = Object.keys(rows[0]);
  let html = '<div style="overflow:auto;max-height:380px"><table><thead><tr>';
  html += cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  rows.slice(0,limit).forEach(r=>{
    html += '<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>';
  });
  html += '</tbody></table></div>'; return html;
}

// === LOAD DATA ===
async function loadData(){
  const trainFile = byId('train-file').files[0];
  const testFile = byId('test-file').files[0];
  if(!trainFile||!testFile){alert('Upload BOTH train and test files');return;}
  info('Loading CSV files…');
  try{
    const trainParsed = Papa.parse(await readFile(trainFile),{header:true,skipEmptyLines:true});
    const testParsed  = Papa.parse(await readFile(testFile),{header:true,skipEmptyLines:true});
    rawTrain = trainParsed.data.map(normalizeTelcoRow);
    rawTest  = testParsed.data.map(normalizeTelcoRow);
    const n = rawTrain.length;
    const churnRate = (100*rawTrain.filter(r=>r.Churn===1).length/n).toFixed(2);
    info(`✅ Loaded Train:${n}, Test:${rawTest.length}, Churn:${churnRate}%`);
    byId('data-preview').innerHTML = headTable(rawTrain,10);
    byId('create-model-btn').disabled=false;
  }catch(e){console.error(e);info(`❌ ${e.message}`);}
}

function computeCorrelation(rows, cols) {
  const corr = {};
  cols.forEach(a => {
    corr[a] = {};
    cols.forEach(b => {
      const x = rows.map(r => parseFloat(r[a])).filter(v => !isNaN(v));
      const y = rows.map(r => parseFloat(r[b])).filter(v => !isNaN(v));
      const n = Math.min(x.length, y.length);
      if (n === 0) corr[a][b] = NaN;
      else {
        const mx = x.reduce((s, v) => s + v, 0) / n;
        const my = y.reduce((s, v) => s + v, 0) / n;
        const num = x.map((v, i) => (v - mx) * (y[i] - my)).reduce((a, b) => a + b, 0);
        const den = Math.sqrt(
          x.map(v => (v - mx) ** 2).reduce((a, b) => a + b, 0) *
          y.map(v => (v - my) ** 2).reduce((a, b) => a + b, 0)
        );
        corr[a][b] = den ? num / den : NaN;
      }
    });
  });
  return corr;
}


// === EDA ===
function runEDA() {
  if (!rawTrain || rawTrain.length === 0) {
    alert("Please load train data first!");
    return;
  }

  info("Running EDA...");

  const container = byId("data-preview");
  container.innerHTML = "<h3>Data Preview</h3>" + headTable(rawTrain, 10);

  const sample = rawTrain[0];

  // === Определяем типы данных ===
  const columns = Object.keys(sample);
  const dataTypes = columns.map(col => {
    const vals = rawTrain.map(r => r[col]);
    const nonNull = vals.find(v => v !== null && v !== undefined && v !== "");
    let dtype = typeof nonNull;
    if (!isNaN(parseFloat(nonNull)) && nonNull !== "" && nonNull !== null) dtype = "number";
    else if (["yes", "no", "male", "female"].includes(String(nonNull).toLowerCase())) dtype = "category";
    else if (String(nonNull).length > 30) dtype = "text";
    return { col, dtype };
  });

  let dtypeHTML = "<h3>Data Types Overview</h3><table><tr><th>Feature</th><th>Detected Type</th></tr>";
  dataTypes.forEach(d => {
    dtypeHTML += `<tr><td>${d.col}</td><td>${d.dtype}</td></tr>`;
  });
  dtypeHTML += "</table>";

  // === Определяем числовые признаки ===
  const numericCols = columns.filter(
    key => !isNaN(parseFloat(sample[key])) && sample[key] !== "" && sample[key] !== null
  );

  // === Пропуски ===
  let missHTML = "<h3>Missing Values</h3><table><tr><th>Feature</th><th>Missing %</th></tr>";
  columns.forEach(c => {
    const miss = rawTrain.filter(r => r[c] === null || r[c] === "").length;
    const pct = (miss / rawTrain.length * 100).toFixed(1);
    missHTML += `<tr><td>${c}</td><td>${pct}%</td></tr>`;
  });
  missHTML += "</table>";

  // === Корреляционная матрица ===
  const corrMatrix = computeCorrelation(rawTrain, numericCols);
  let corrHTML = "<h3>Correlation Matrix</h3><table><tr><th></th>";
  numericCols.forEach(c => (corrHTML += `<th>${c}</th>`));
  corrHTML += "</tr>";
  numericCols.forEach(a => {
    corrHTML += `<tr><th>${a}</th>`;
    numericCols.forEach(b => {
      const v = corrMatrix[a][b];
      const color = isNaN(v)
        ? "transparent"
        : v > 0
          ? `rgba(56,189,248,${Math.abs(v)})`
          : `rgba(239,68,68,${Math.abs(v)})`;
      corrHTML += `<td style="background:${color};color:#fff;text-align:center;">${
        isNaN(v) ? "" : v.toFixed(2)
      }</td>`;
    });
    corrHTML += "</tr>";
  });
  corrHTML += "</table>";

  // === Numeric summary ===
  let numHTML = "<h3>Numeric Summary</h3><table><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>";
  numericCols.forEach(c => {
    const vals = rawTrain.map(r => parseFloat(r[c])).filter(v => !isNaN(v));
    if (vals.length === 0) return;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / vals.length);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    numHTML += `<tr><td>${c}</td><td>${mean.toFixed(2)}</td><td>${std.toFixed(2)}</td><td>${min.toFixed(2)}</td><td>${max.toFixed(2)}</td></tr>`;
  });
  numHTML += "</table>";

  // === Churn distribution ===
  const yes = rawTrain.filter(r => r.Churn === 1).length;
  const no = rawTrain.length - yes;
  const chartHTML = `
    <h3>Churn Distribution</h3>
    <div style="max-width:400px;height:250px;">
      <canvas id="churnChart"></canvas>
    </div>
  `;

  // === Итоговый layout ===
  container.innerHTML += `
    <div style="margin-top:20px;">${dtypeHTML}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;">
      <div>${corrHTML}</div>
      <div>${missHTML}</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;">
      <div>${numHTML}</div>
      <div>${chartHTML}</div>
    </div>
  `;

  // === График ===
  const ctx = document.getElementById("churnChart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["No", "Yes"],
      datasets: [{
        label: "Churn Count",
        data: [no, yes],
        backgroundColor: ["#22c55e", "#ef4444"]
      }]
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });
}


// === FEATURE ENCODING ===
function preprocess(rows){
  const num=['tenure','MonthlyCharges','TotalCharges'];
  const cat=['Contract','InternetService','PaymentMethod','gender','SeniorCitizen','Partner','Dependents','PaperlessBilling'];
  const cats={};
  cat.forEach(c=>cats[c]=[...new Set(rows.map(r=>r[c]))]);
  const features=rows.map(r=>{
    const nums=num.map(f=>parseFloat(r[f]||0));
    const enc=[];
    cat.forEach(c=>{
      cats[c].forEach(v=>enc.push(r[c]===v?1:0));
    });
    return nums.concat(enc);
  });
  const X=tf.tensor2d(features);
  const y=rows.map(r=>r.Churn); 
  return {X,y:tf.tensor1d(y)};
}

// === CREATE MODEL ===
function createModel(){
  const inputSize = preprocess(rawTrain).X.shape[1];
  model=tf.sequential();
  model.add(tf.layers.dense({inputShape:[inputSize],units:32,activation:'relu'}));
  model.add(tf.layers.dropout({rate:0.3}));
  model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  model.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  info('✅ Model created');
  byId('train-btn').disabled=false;
}

// === TRAIN MODEL ===
async function trainModel(){
  info('Training...');
  const {X,y}=preprocess(rawTrain);
  const split=Math.floor(X.shape[0]*0.8);
  const trainX=X.slice(0,split),trainY=y.slice(0,split);
  valData=X.slice(split),valLabels=y.slice(split);
  await model.fit(trainX,trainY,{
    epochs:20,batchSize:32,validationData:[valData,valLabels],
    callbacks:tfvis.show.fitCallbacks(
      {name:'Training Performance',tab:'Training'},
      ['loss','acc','val_loss','val_acc'],
      {callbacks:['onEpochEnd']}
    )
  });
  info('✅ Training complete');
  validationPreds=model.predict(valData);
  byId('threshold-slider').disabled=false;
  byId('threshold-slider').addEventListener('input',updateMetrics);
  updateMetrics();
  byId('predict-btn').disabled=false;
}

// === METRICS ===
async function updateMetrics(){
  const thr=parseFloat(byId('threshold-slider').value);
  byId('threshold-value').textContent=thr.toFixed(2);
  const yTrue=valLabels.arraySync(),yPred=validationPreds.arraySync().map(p=>p[0]);
  let tp=0,tn=0,fp=0,fn=0;
  for(let i=0;i<yPred.length;i++){
    const p=yPred[i]>=thr?1:0,a=yTrue[i];
    if(p===1&&a===1)tp++;else if(p===0&&a===0)tn++;
    else if(p===1&&a===0)fp++;else fn++;
  }
  const precision=tp/(tp+fp)||0,recall=tp/(tp+fn)||0;
  const f1=2*(precision*recall)/(precision+recall)||0;
  const acc=(tp+tn)/(tp+tn+fp+fn);
  byId('confusion-matrix').innerHTML=
    `<table><tr><th></th><th>Pred +</th><th>Pred -</th></tr>
    <tr><th>Actual +</th><td>${tp}</td><td>${fn}</td></tr>
    <tr><th>Actual -</th><td>${fp}</td><td>${tn}</td></tr></table>`;
  byId('performance-metrics').innerHTML=
    `<p>Accuracy: ${(acc*100).toFixed(2)}%</p>
     <p>Precision: ${precision.toFixed(3)}</p>
     <p>Recall: ${recall.toFixed(3)}</p>
     <p>F1: ${f1.toFixed(3)}</p>`;
  await plotROC(yTrue,yPred);
}
async function plotROC(yTrue,yPred){
  const steps=50,data=[];
  for(let t=0;t<=1;t+=1/steps){
    let tp=0,fp=0,fn=0,tn=0;
    for(let i=0;i<yPred.length;i++){
      const p=yPred[i]>=t?1:0,a=yTrue[i];
      if(a===1&&p===1)tp++; if(a===0&&p===1)fp++;
      if(a===1&&p===0)fn++; if(a===0&&p===0)tn++;
    }
    const tpr=tp/(tp+fn)||0,fpr=fp/(fp+tn)||0;
    data.push({x:fpr,y:tpr});
  }
  tfvis.render.linechart({name:'ROC Curve',tab:'Evaluation'},
    {values:data,series:['ROC']},{xLabel:'FPR',yLabel:'TPR'});
}

// === PREDICT ON TEST ===
async function predict(){
  const {X}=preprocess(rawTest);
  testPreds=model.predict(X).arraySync().map(a=>a[0]);
  const res=rawTest.map((r,i)=>({
    customerID:r.customerID,
    Probability:testPreds[i].toFixed(4),
    Predicted:testPreds[i]>=0.5?'Churn':'Stay'
  }));
  const html=headTable(res,10);
  byId('prediction-output').innerHTML=html;
  byId('export-btn').disabled=false;
}

// === EXPORT CSV ===
function exportCSV(){
  let csv='customerID,Probability,Prediction\n';
  rawTest.forEach((r,i)=>{
    csv+=`${r.customerID},${testPreds[i].toFixed(4)},${testPreds[i]>=0.5?'Churn':'Stay'}\n`;
  });
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='predictions.csv';a.click();
}

// === EVENTS ===
byId('load-data-btn').addEventListener('click',loadData);
byId('eda-btn').addEventListener('click',runEDA);
byId('create-model-btn').addEventListener('click',createModel);
byId('train-btn').addEventListener('click',trainModel);
byId('predict-btn').addEventListener('click',predict);
byId('export-btn').addEventListener('click',exportCSV);
