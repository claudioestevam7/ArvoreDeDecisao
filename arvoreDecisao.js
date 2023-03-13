const DTC = require('decision-tree-classifier');
const iris = require('ml-dataset-iris');

const data = iris.getNumbers();
const target = iris.getLabels();
const [X_train, X_test, y_train, y_test] = splitData(data, target);

const dtc = new DTC({
  gainFunction: 'gini',
  maxDepth: 3
});
dtc.train(X_train, y_train);

const y_pred = dtc.predict(X_test);
const accuracy = getAccuracy(y_test, y_pred);
console.log(`Acurácia: ${accuracy}`);
