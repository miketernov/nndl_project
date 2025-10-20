// Telco Customer Churn — In‑Browser ML (TensorFlow.js)
// ----------------------------------------------------
// Features implemented:
// 1) Load train/test CSV (or single CSV → split)
// 2) EDA: preview, class balance, numeric correlation, quick bars in tfjs‑vis
// 3) Create model
// 4) Live training monitor (tfjs‑vis)
// 5) Metrics: ROC, confusion matrix, Accuracy/Precision/Recall/F1
// 6) Prediction table with churn probability and expected loss
// 7) Export CSV (predictions + economics)

// =============== Global State ===============
let rawTrain = null;
let rawTest  = null;
let targetCol = 'Churn';
let posLabel  = 'Yes';
let idCol     = 'customerID';

let trainX = null, trainY = null, valX = null, valY = null, testX = null, testIDs = [];
let model = null; let valPreds = null; let testPreds = null;

// Fitted preprocessors
const prep = {
  numCols: ['tenure','MonthlyCharges','TotalCharges'],
  catCols: [
    'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'
  ],
  // stats learned from train
  numMean: {}, numStd: {}, catLevels: {},
};

// =============== Helpers ===============
const $ = sel => document.querySelector(sel);
const byId = id => document.getElementById(id);

function info(msg){ const el = byId('data-status'); if (el) el.innerHTML = msg; }

function readFile(file){
  return new Promise((resolve, reject)=>{
    const r = new FileReader();
    r.onload = e => resolve(e.target.result);
    r.onerror = () => reject(new Error('Failed to read file'));
    r.readAsText(file);
  });
}

function parseCSV(text){
  // simple robust CSV → array of objects
  const rows = []; let i=0, field='', row=[], inQ=false; if (text.charCodeAt(0)===0xFEFF) text = text.slice(1);
  const pushF=()=>{ row.push(inQ?field.replace(/""/g,'"'):field); field=''; };
  const pushR=()=>{ rows.push(row); row=[]; };
  while(i<text.length){ const ch=text[i];
    if(inQ){ if(ch==='"'){ if(text[i+1]==='"'){ field+='"'; i++; } else inQ=false; } else field+=ch; }
    else { if(ch==='"') inQ=true; else if(ch===',') pushF(); else if(ch==='\n'){ pushF(); pushR(); } else if(ch!=='\r') field+=ch; }
    i++;
  }
  if(field.length||row.length){ pushF(); pushR(); }
  const headers = rows.shift().map(h=> String(h||'').trim());
  return rows.filter(r=> r.length && r.some(v=> v!==''))
    .map(r=>{ const o={}; headers.forEach((h,j)=>{ let v=r[j]; if(v==='') v=null; o[h]=v; }); return o; });
}

function toNumberSafe(v){ if(v===null||v===undefined||v==='') return null; const n=Number(v); return Number.isFinite(n)?n:null; }

function normalizeTelcoRow(o){
  const r = {...o};
  // strip spaces
  Object.keys(r).forEach(k=>{ if(typeof r[k]==='string') r[k]=r[k].trim(); });
  // fix TotalCharges often being " "
  r.TotalCharges = toNumberSafe(r.TotalCharges);
  r.tenure = toNumberSafe(r.tenure);
  r.MonthlyCharges = toNumberSafe(r.MonthlyCharges);
  // Lowercase some
  if(r.gender!=null) r.gender = String(r.gender).toLowerCase();
  // unify Yes/No family; treat "No internet service" as "No" for relevant cols
  const ynCols = ['Partner','Dependents','PhoneService','PaperlessBilling'];
  ynCols.forEach(c=>{ if(r[c]!=null){ const s=String(r[c]).toLowerCase(); r[c] = (s==='yes'||s==='y'||s==='1')? 'Yes':'No'; }});
  const netDep = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines'];
  netDep.forEach(c=>{ if(r[c]!=null){ let s=String(r[c]).toLowerCase(); if(s==='no internet service'||s==='no phone service') s='no'; r[c] = (s==='yes')?'Yes':'No'; }});
  // SeniorCitizen numeric 0/1
  if(r.SeniorCitizen!=null) r.SeniorCitizen = String(r.SeniorCitizen)==='1'?'Yes':'No';
  // Target to 0/1 (keep original string too)
  if(r[targetCol]!=null){ const s=String(r[targetCol]).trim(); r[targetCol] = (s===posLabel||s==='1')?1:0; }
  return r;
}

function headTable(rows, limit=10){
  if(!rows||!rows.length){ return '<p class="small">No data.</p>'; }
  const cols = Object.keys(rows[0]);
  let html = '<div style="overflow:auto;max-height:380px"><table><thead><tr>'+
    cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  rows.slice(0,limit).forEach(r=>{
    html += '<tr>'+cols.map(c=>`<td>${r[c]===undefined||r[c]===null?'NULL':r[c]}</td>`).join('')+'</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

function unique(arr){ return Array.from(new Set(arr)); }

function isNumericCol(data, col){
  return data.every(r => r[col]===null||r[col]===undefined || !isNaN(Number(r[col])));
}

function corr(a,b){
  const n=a.length; if(!n) return 0; const ma=a.reduce((s,x)=>s+x,0)/n; const mb=b.reduce((s,x)=>s+x,0)/n;
  let num=0, da=0, db=0; for(let i=0;i<n;i++){ const xa=a[i]-ma, xb=b[i]-mb; num+=xa*xb; da+=xa*xa; db+=xb*xb; }
  const den=Math.sqrt(da*db); return den? num/den : 0;
}

function renderCorrTable(data){
  // numeric-only columns
  const cols = Object.keys(data[0]).filter(c=> isNumericCol(data,c) && c!==targetCol);
  const mat = cols.map(()=> Array(cols.length).fill(0));
  const arrays = cols.map(c=> data.map(r=> Number(r[c]||0)));
  for(let i=0;i<cols.length;i++) for(let j=i;j<cols.length;j++){
    const v=corr(arrays[i], arrays[j]); mat[i][j]=mat[j][i]=v;
  }
  // build table with background heat
  let html = '<div style="overflow:auto;max-height:380px"><table class="heat"><thead><tr><th></th>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  for(let i=0;i<cols.length;i++){
    html += `<tr><th>${cols[i]}</th>`;
    for(let j=0;j<cols.length;j++){
      const v=mat[i][j]; const c = v>=0? `rgba(34,197,94,${Math.abs(v)})` : `rgba(239,68,68,${Math.abs(v)})`;
      html += `<td style="--bgHeat:${c}"><span>${v.toFixed(2)}</span></td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  byId('corr-table').innerHTML = html;
}

function tfvisBar(name, pairs){
  const data = pairs.map(([x,y])=>({x:String(x), y:Number(y)}));
  tfvis.render.barchart({ name, tab:'Charts' }, data, { xLabel:'', yLabel:'% / count' });
  byId('charts-note').innerHTML = 'Open the tfjs‑vis visor (bottom‑right) → Charts tab to see bar plots.';
}

// =============== Load & EDA ===============
byId('load-data-btn').addEventListener('click', loadData);
byId('btnClear').addEventListener('click', ()=> location.reload());

async function loadData() {
  const trainFile = byId('train-file').files[0];
  const testFile = byId('test-file').files[0];

  if (!trainFile || !testFile) {
    alert('Please upload BOTH train.csv and test.csv files.');
    return;
  }

  info('Loading CSV files…');

  try {
    // === TRAIN ===
    const trainText = await readFile(trainFile);
    const trainParsed = Papa.parse(trainText, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true
    });
    rawTrain = trainParsed.data.map(normalizeTelcoRow);

    // === TEST ===
    const testText = await readFile(testFile);
    const testParsed = Papa.parse(testText, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true
    });
    rawTest = testParsed.data.map(normalizeTelcoRow);

    // === EDA summary ===
    const nTrain = rawTrain.length;
    const nCols = Object.keys(rawTrain[0] || {}).length;
    const posCnt = rawTrain.filter(r => r.Churn === 1).length;
    const churnRate = (posCnt / nTrain * 100).toFixed(2);

    byId('data-status').innerHTML = `✅ Loaded: Train ${nTrain} rows × ${nCols} cols, Test ${rawTest.length} rows`;
    byId('shapeTrain').textContent = `train: ${nTrain}`;
    byId('shapeTest').textContent = `test: ${rawTest.length}`;
    byId('churnRate').textContent = `churn: ${churnRate}%`;

    // Show preview
    byId('data-preview').innerHTML = headTable(rawTrain, 10);

    // Enable the next button
    byId('create-model-btn').disabled = false;

  } catch (err) {
    console.error(err);
    byId('data-status').innerHTML = `❌ Error loading data: ${err.message}`;
  }
}

// =============== Preprocess ===============
function fitPreprocess(){
  const T = rawTrain;
  // numeric stats
  prep.numCols.forEach(c=>{
    const arr = T.map(r=> toNumberSafe(r[c])).map(v=> v??0);
    const mean = arr.reduce((s,x)=>s+x,0)/arr.length;
    const std  = Math.sqrt(arr.reduce((s,x)=> s+(x-mean)*(x-mean),0)/arr.length)||1;
    prep.numMean[c]=mean; prep.numStd[c]=std;
  });
  // categorical levels from train
  prep.catCols.forEach(c=>{
    prep.catLevels[c] = unique(T.map(r=> r[c]==null?'NA':String(r[c])));
  });
}

function transformRow(r){
  // numeric z‑score
  const num = prep.numCols.map(c=>{ const v=toNumberSafe(r[c])??0; return (v-prep.numMean[c])/prep.numStd[c]; });
  // one‑hot cats in fixed order
  const cats = prep.catCols.flatMap(c=>{
    const levels = prep.catLevels[c];
    const val = r[c]==null?'NA':String(r[c]);
    return levels.map(l=> l===val?1:0);
  });
  return num.concat(cats);
}

function makeTensors(){
  fitPreprocess();
  const X = rawTrain.map(transformRow);
  const y = rawTrain.map(r=> r[targetCol]);
  // train/val split
  const cut = Math.floor(X.length*0.8);
  trainX = tf.tensor2d(X.slice(0,cut));
  trainY = tf.tensor1d(y.slice(0,cut));
  valX   = tf.tensor2d(X.slice(cut));
  valY   = tf.tensor1d(y.slice(cut));
  // test
  testIDs = rawTest.map(r=> r[idCol]||r.customerID||r.CustomerID||r.id||r.ID);
  const Xtest = rawTest.map(transformRow);
  testX  = tf.tensor2d(Xtest);
}

// =============== Model ===============
byId('create-model-btn').addEventListener('click', ()=>{
  makeTensors();
  const hid = parseInt(byId('hid').value||'32',10);
  model = tf.sequential();
  model.add(tf.layers.dense({units:hid, activation:'relu', inputShape:[trainX.shape[1]]}));
  model.add(tf.layers.dropout({rate:0.2}));
  model.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  model.compile({ optimizer: tf.train.adam(0.001), loss:'binaryCrossentropy', metrics:['accuracy'] });
  const txt = `Model: [${trainX.shape[1]}] → Dense(${hid}) → Dropout(.2) → Dense(1,sigmoid). Params: ${model.countParams()}`;
  byId('training-status').innerText = txt;
  byId('train-btn').disabled = false;
});

byId('train-btn').addEventListener('click', async ()=>{
  const epochs = parseInt(byId('epochs').value||'20',10);
  const batch  = parseInt(byId('batch').value||'64',10);
  byId('training-status').innerText = 'Training…';

  const visorCb = tfvis.show.fitCallbacks(
    { name:'Training Performance' }, ['loss','acc','val_loss','val_acc'], { callbacks:['onEpochEnd'] }
  );

  await model.fit(trainX, trainY, {
    epochs, batchSize: batch, validationData:[valX,valY],
    callbacks: {
      onEpochEnd: async (ep, logs)=>{
        byId('training-status').innerText = `Epoch ${ep+1}/${epochs} — loss:${logs.loss.toFixed(4)} acc:${logs.acc.toFixed(3)} vloss:${logs.val_loss.toFixed(4)} vacc:${logs.val_acc.toFixed(3)}`;
        await visorCb.onEpochEnd(ep, logs);
      }
    }
  });

  // post‑train
  valPreds = model.predict(valX).arraySync().map(v=> Array.isArray(v)? v[0]:v);
  byId('threshold-slider').disabled = false;
  byId('predict-btn').disabled = false;
  updateMetrics();
});

// =============== Metrics ===============
byId('threshold-slider').addEventListener('input', updateMetrics);

function updateMetrics(){
  if(!valPreds) return; const thr = parseFloat(byId('threshold-slider').value); byId('threshold-value').innerText = thr.toFixed(2);
  const yTrue = valY.arraySync();
  let tp=0,tn=0,fp=0,fn=0; for(let i=0;i<valPreds.length;i++){ const p = valPreds[i]>=thr?1:0; const y=yTrue[i]; if(p===1&&y===1) tp++; else if(p===0&&y===0) tn++; else if(p===1&&y===0) fp++; else fn++; }
  // Confusion matrix table
  byId('confusion-matrix').innerHTML = `
    <table>
      <tr><th></th><th>Pred +</th><th>Pred −</th></tr>
      <tr><th>Actual +</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual −</th><td>${fp}</td><td>${tn}</td></tr>
    </table>`;
  const precision = tp/(tp+fp)||0, recall = tp/(tp+fn)||0, f1 = 2*precision*recall/(precision+recall)||0, acc=(tp+tn)/(tp+tn+fp+fn)||0;
  byId('performance-metrics').innerHTML = `
    <p>Accuracy: ${(acc*100).toFixed(2)}%</p>
    <p>Precision: ${precision.toFixed(3)}</p>
    <p>Recall: ${recall.toFixed(3)}</p>
    <p>F1: ${f1.toFixed(3)}</p>`;
  // ROC & AUC in visor
  plotROC(yTrue, valPreds);
}

function plotROC(yTrue, yProb){
  const ths = Array.from({length:101}, (_,i)=> i/100);
  const points = ths.map(t=>{ let tp=0,fp=0,tn=0,fn=0; for(let i=0;i<yProb.length;i++){ const p=yProb[i]>=t?1:0, y=yTrue[i]; if(y===1){ if(p===1) tp++; else fn++; } else { if(p===1) fp++; else tn++; } } return { x: fp/(fp+tn)||0, y: tp/(tp+fn)||0 }; });
  // AUC (trapezoid)
  let auc=0; for(let i=1;i<points.length;i++){ const w=points[i].x-points[i-1].x; const h=(points[i].y+points[i-1].y)/2; auc+=w*h; }
  tfvis.render.linechart({ name:'ROC Curve', tab:'Evaluation' }, { values: points }, { xLabel:'FPR', yLabel:'TPR' });
  const m = byId('performance-metrics'); m.innerHTML += `<p>AUC: ${Math.abs(auc).toFixed(3)}</p>`;
}

// =============== Prediction & Export ===============
byId('predict-btn').addEventListener('click', predict);
byId('export-btn').addEventListener('click', exportCSV);

function predict(){
  const thr = parseFloat(byId('threshold').value||'0.35');
  const ltv = parseFloat(byId('ltv').value||'800');
  const offer = parseFloat(byId('offer').value||'20');
  const probs = model.predict(testX).arraySync().map(v=> Array.isArray(v)? v[0]:v);
  testPreds = probs;
  const rows = probs.map((p,i)=>{
    const pred = p>=thr?1:0; const expected = pred? (ltv - offer): 0; // simple value
    return { customerID: testIDs[i], churn_pred: pred, churn_proba: p, expected_value: expected };
  });
  // render preview
  const cols=['customerID','churn_pred','churn_proba','expected_value'];
  let html='<div style="overflow:auto;max-height:380px"><table><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  rows.slice(0,20).forEach(r=>{ html+='<tr>'+cols.map(c=>`<td>${c==='churn_proba'||c==='expected_value'? Number(r[c]).toFixed(4): r[c]}</td>`).join('')+'</tr>'; });
  html+='</tbody></table></div>';
  byId('prediction-output').innerHTML = html + `<p class="small">Total: ${rows.length}. Use Export to download full CSV.</p>`;
  byId('export-btn').disabled = false;
}

function exportCSV(){
  if(!testPreds) { alert('Make predictions first.'); return; }
  const thr = parseFloat(byId('threshold').value||'0.35');
  const ltv = parseFloat(byId('ltv').value||'800');
  const offer = parseFloat(byId('offer').value||'20');
  let csv = 'customerID,churn_pred,churn_proba,expected_value\n';
  testPreds.forEach((p,i)=>{ const pred=p>=thr?1:0; const expected=pred?(ltv-offer):0; csv += `${testIDs[i]},${pred},${p.toFixed(6)},${expected.toFixed(2)}\n`; });
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download='telco_churn_predictions.csv'; a.click();
}
