(function () {
  'use strict';

  var GaussianNB = ML.NaiveBayes.GaussianNB;
  var RandomForestClassifier = ML.RandomForestClassifier;
  var ConfusionMatrix = ML.ConfusionMatrix;

  // Application state
  var state = {
    featureNames: [],
    features: [], // number[][]
    labels: [], // string[]
    classes: [], // unique label names
    models: null // { nb, rf } after training
  };

  // ---------- Sample dataset (Iris subset, 30 rows per class) ----------
  var IRIS_CSV =
    'sepal_length,sepal_width,petal_length,petal_width,species\n' +
    '5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa\n' +
    '4.6,3.1,1.5,0.2,setosa\n5.0,3.6,1.4,0.2,setosa\n5.4,3.9,1.7,0.4,setosa\n' +
    '4.6,3.4,1.4,0.3,setosa\n5.0,3.4,1.5,0.2,setosa\n4.4,2.9,1.4,0.2,setosa\n' +
    '4.9,3.1,1.5,0.1,setosa\n5.4,3.7,1.5,0.2,setosa\n4.8,3.4,1.6,0.2,setosa\n' +
    '4.8,3.0,1.4,0.1,setosa\n4.3,3.0,1.1,0.1,setosa\n5.8,4.0,1.2,0.2,setosa\n' +
    '5.7,4.4,1.5,0.4,setosa\n5.4,3.9,1.3,0.4,setosa\n5.1,3.5,1.4,0.3,setosa\n' +
    '5.7,3.8,1.7,0.3,setosa\n5.1,3.8,1.5,0.3,setosa\n5.4,3.4,1.7,0.2,setosa\n' +
    '5.1,3.7,1.5,0.4,setosa\n4.6,3.6,1.0,0.2,setosa\n5.1,3.3,1.7,0.5,setosa\n' +
    '4.8,3.4,1.9,0.2,setosa\n5.0,3.0,1.6,0.2,setosa\n5.0,3.4,1.6,0.4,setosa\n' +
    '5.2,3.5,1.5,0.2,setosa\n5.2,3.4,1.4,0.2,setosa\n4.7,3.2,1.6,0.2,setosa\n' +
    '7.0,3.2,4.7,1.4,versicolor\n6.4,3.2,4.5,1.5,versicolor\n6.9,3.1,4.9,1.5,versicolor\n' +
    '5.5,2.3,4.0,1.3,versicolor\n6.5,2.8,4.6,1.5,versicolor\n5.7,2.8,4.5,1.3,versicolor\n' +
    '6.3,3.3,4.7,1.6,versicolor\n4.9,2.4,3.3,1.0,versicolor\n6.6,2.9,4.6,1.3,versicolor\n' +
    '5.2,2.7,3.9,1.4,versicolor\n5.0,2.0,3.5,1.0,versicolor\n5.9,3.0,4.2,1.5,versicolor\n' +
    '6.0,2.2,4.0,1.0,versicolor\n6.1,2.9,4.7,1.4,versicolor\n5.6,2.9,3.6,1.3,versicolor\n' +
    '6.7,3.1,4.4,1.4,versicolor\n5.6,3.0,4.5,1.5,versicolor\n5.8,2.7,4.1,1.0,versicolor\n' +
    '6.2,2.2,4.5,1.5,versicolor\n5.6,2.5,3.9,1.1,versicolor\n5.9,3.2,4.8,1.8,versicolor\n' +
    '6.1,2.8,4.0,1.3,versicolor\n6.3,2.5,4.9,1.5,versicolor\n6.1,2.8,4.7,1.2,versicolor\n' +
    '6.4,2.9,4.3,1.3,versicolor\n6.6,3.0,4.4,1.4,versicolor\n6.8,2.8,4.8,1.4,versicolor\n' +
    '6.7,3.0,5.0,1.7,versicolor\n6.0,2.9,4.5,1.5,versicolor\n5.7,2.6,3.5,1.0,versicolor\n' +
    '6.3,3.3,6.0,2.5,virginica\n5.8,2.7,5.1,1.9,virginica\n7.1,3.0,5.9,2.1,virginica\n' +
    '6.3,2.9,5.6,1.8,virginica\n6.5,3.0,5.8,2.2,virginica\n7.6,3.0,6.6,2.1,virginica\n' +
    '4.9,2.5,4.5,1.7,virginica\n7.3,2.9,6.3,1.8,virginica\n6.7,2.5,5.8,1.8,virginica\n' +
    '7.2,3.6,6.1,2.5,virginica\n6.5,3.2,5.1,2.0,virginica\n6.4,2.7,5.3,1.9,virginica\n' +
    '6.8,3.0,5.5,2.1,virginica\n5.7,2.5,5.0,2.0,virginica\n5.8,2.8,5.1,2.4,virginica\n' +
    '6.4,3.2,5.3,2.3,virginica\n6.5,3.0,5.5,1.8,virginica\n7.7,3.8,6.7,2.2,virginica\n' +
    '7.7,2.6,6.9,2.3,virginica\n6.0,2.2,5.0,1.5,virginica\n6.9,3.2,5.7,2.3,virginica\n' +
    '5.6,2.8,4.9,2.0,virginica\n7.7,2.8,6.7,2.0,virginica\n6.3,2.7,4.9,1.8,virginica\n' +
    '6.7,3.3,5.7,2.1,virginica\n7.2,3.2,6.0,1.8,virginica\n6.2,2.8,4.8,1.8,virginica\n' +
    '6.1,3.0,4.9,1.8,virginica\n6.4,2.8,5.6,2.1,virginica\n7.2,3.0,5.8,1.6,virginica\n';

  // ---------- DOM helpers ----------
  function $(id) { return document.getElementById(id); }
  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }

  // ---------- CSV parsing ----------
  function parseCSV(text) {
    var lines = text.split(/\r?\n/).filter(function (l) { return l.trim() !== ''; });
    if (lines.length < 2) throw new Error('CSV needs a header row and at least one data row.');
    var header = lines[0].split(',').map(function (h) { return h.trim(); });
    if (header.length < 2) throw new Error('CSV needs at least one feature column and a label column.');
    var featureNames = header.slice(0, -1);
    var features = [];
    var labels = [];
    for (var i = 1; i < lines.length; i++) {
      var cells = lines[i].split(',').map(function (c) { return c.trim(); });
      if (cells.length !== header.length) {
        throw new Error('Row ' + (i + 1) + ' has ' + cells.length + ' columns, expected ' + header.length + '.');
      }
      var row = [];
      for (var j = 0; j < featureNames.length; j++) {
        var v = Number(cells[j]);
        if (!isFinite(v)) {
          throw new Error('Row ' + (i + 1) + ', column "' + featureNames[j] + '" is not numeric: "' + cells[j] + '".');
        }
        row.push(v);
      }
      features.push(row);
      labels.push(cells[cells.length - 1]);
    }
    return { featureNames: featureNames, features: features, labels: labels };
  }

  // ---------- Dataset loading ----------
  function setDataset(data) {
    state.featureNames = data.featureNames;
    state.features = data.features;
    state.labels = data.labels;
    state.classes = Array.from(new Set(data.labels));
    state.models = null;

    var info = $('datasetInfo');
    info.textContent = state.features.length + ' samples, ' +
      state.featureNames.length + ' features, ' +
      state.classes.length + ' classes (' + state.classes.join(', ') + ')';
    show(info);
    renderPreview();
    $('trainBtn').disabled = false;
    hide($('resultsCard'));
    hide($('predictCard'));
  }

  function renderPreview() {
    var table = $('previewTable');
    table.innerHTML = '';
    var thead = document.createElement('thead');
    var trh = document.createElement('tr');
    state.featureNames.concat(['label']).forEach(function (name) {
      var th = document.createElement('th');
      th.textContent = name;
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var n = Math.min(5, state.features.length);
    for (var i = 0; i < n; i++) {
      var tr = document.createElement('tr');
      state.features[i].concat([state.labels[i]]).forEach(function (val) {
        var td = document.createElement('td');
        td.textContent = String(val);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    show($('previewWrap'));
  }

  // ---------- Train / validate ----------
  function shuffleIndices(n) {
    var idx = [];
    for (var i = 0; i < n; i++) idx.push(i);
    for (var k = n - 1; k > 0; k--) {
      var r = Math.floor(Math.random() * (k + 1));
      var tmp = idx[k]; idx[k] = idx[r]; idx[r] = tmp;
    }
    return idx;
  }

  function trainAndValidate() {
    var testRatio = Number($('testSplit').value);
    var nEstimators = Math.max(1, Math.min(200, Number($('nEstimators').value) || 25));
    var doShuffle = $('shuffleData').checked;

    var n = state.features.length;
    var order = doShuffle ? shuffleIndices(n) : Array.from({ length: n }, function (_, i) { return i; });

    var nTest = Math.max(1, Math.round(n * testRatio));
    var nTrain = n - nTest;
    if (nTrain < state.classes.length) {
      alert('Not enough training samples for the chosen split.');
      return;
    }

    var trainX = [], trainYIdx = [], testX = [], testY = [];
    order.forEach(function (idx, pos) {
      var classIdx = state.classes.indexOf(state.labels[idx]);
      if (pos < nTrain) {
        trainX.push(state.features[idx]);
        trainYIdx.push(classIdx);
      } else {
        testX.push(state.features[idx]);
        testY.push(state.labels[idx]);
      }
    });

    var nb, nbPred, rf, rfPred;
    try {
      // Gaussian Naive Bayes
      nb = new GaussianNB();
      nb.train(trainX, trainYIdx);
      nbPred = nb.predict(testX).map(function (i) { return state.classes[i]; });

      // Random Forest
      rf = new RandomForestClassifier({ nEstimators: nEstimators, seed: 42 });
      rf.train(trainX, trainYIdx);
      rfPred = rf.predict(testX).map(function (i) { return state.classes[i]; });
    } catch (err) {
      alert('Training failed: ' + err.message +
        '\nTry a larger dataset or a smaller test split.');
      return;
    }

    state.models = { nb: nb, rf: rf };

    renderResult('nbResult', testY, nbPred);
    renderResult('rfResult', testY, rfPred);
    show($('resultsCard'));
    buildPredictForm();
    show($('predictCard'));
  }

  function renderResult(containerId, actual, predicted) {
    var container = $(containerId);
    var cm = ConfusionMatrix.fromLabels(actual, predicted);
    container.querySelector('.accuracy').textContent =
      'Accuracy: ' + (cm.getAccuracy() * 100).toFixed(1) + '% (' + actual.length + ' test samples)';
    renderConfusionMatrix(container.querySelector('table.confusion'), cm);
  }

  function renderConfusionMatrix(table, cm) {
    var labels = cm.getLabels();
    var matrix = cm.getMatrix();
    table.innerHTML = '';

    var thead = document.createElement('thead');
    var trh = document.createElement('tr');
    var corner = document.createElement('th');
    corner.textContent = 'actual \\ predicted';
    trh.appendChild(corner);
    labels.forEach(function (l) {
      var th = document.createElement('th');
      th.textContent = String(l);
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    matrix.forEach(function (row, i) {
      var tr = document.createElement('tr');
      var th = document.createElement('th');
      th.textContent = String(labels[i]);
      tr.appendChild(th);
      row.forEach(function (count, j) {
        var td = document.createElement('td');
        td.textContent = String(count);
        if (i === j && count > 0) td.classList.add('diag');
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  // ---------- Prediction ----------
  function buildPredictForm() {
    var wrap = $('predictInputs');
    wrap.innerHTML = '';
    state.featureNames.forEach(function (name, i) {
      var label = document.createElement('label');
      label.textContent = name;
      var input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.dataset.featureIndex = String(i);
      input.value = String(state.features[0][i]);
      label.appendChild(input);
      wrap.appendChild(label);
    });
    hide($('predictOutput'));
  }

  function predict() {
    if (!state.models) return;
    var inputs = $('predictInputs').querySelectorAll('input');
    var sample = [];
    for (var i = 0; i < inputs.length; i++) {
      var v = Number(inputs[i].value);
      if (!isFinite(v)) {
        alert('Please enter a numeric value for "' + state.featureNames[i] + '".');
        return;
      }
      sample.push(v);
    }
    var nbClass = state.classes[state.models.nb.predict([sample])[0]];
    var rfClass = state.classes[state.models.rf.predict([sample])[0]];
    var out = $('predictOutput');
    out.textContent = 'Naive Bayes: ' + nbClass + '  |  Random Forest: ' + rfClass;
    show(out);
  }

  // ---------- Wire up events ----------
  $('loadIrisBtn').addEventListener('click', function () {
    setDataset(parseCSV(IRIS_CSV));
  });

  $('csvInput').addEventListener('change', function (e) {
    var file = e.target.files && e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      try {
        setDataset(parseCSV(String(reader.result)));
      } catch (err) {
        alert('Could not parse CSV: ' + err.message);
      }
    };
    reader.readAsText(file);
  });

  $('trainBtn').addEventListener('click', trainAndValidate);
  $('predictBtn').addEventListener('click', predict);
})();
