// Importe la bibliothèque TensorFlow.js pour Node.js
const tf = require('@tensorflow/tfjs-node');

// Crée un modèle séquentiel
const model = tf.sequential();

// Ajoute une couche d'embedding avec une taille d'entrée de 4, une taille de sortie de 4 et une longueur d'entrée de 1
model.add(tf.layers.embedding({ inputDim: 4, outputDim: 4, inputLength: 1 }));

// Ajoute une couche dense avec 10 unités et une activation 'relu'
model.add(tf.layers.dense({ units: 10, activation: 'relu' }));

// Ajoute une autre couche dense avec 5 unités et une activation 'relu'
model.add(tf.layers.dense({ units: 5, activation: 'relu' }));

// Ajoute une couche dense finale avec 3 unités et une activation 'softmax'
model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

// Aplatit la sortie pour correspondre à la forme des étiquettes
model.add(tf.layers.flatten());

// Compile le modèle en utilisant la fonction de perte de l'entropie croisée catégorique et l'optimiseur Stochastic Gradient Descent (SGD)
model.compile({ loss: 'categoricalCrossentropy', optimizer: 'sgd' });

// Crée un dictionnaire pour mapper les chaînes de caractères aux indices
const inputMapping = { 'cat': 1, 'dog': 2, 'bird': 3 };

// Crée un tenseur 2D pour les entrées (x) en utilisant les indices correspondants
const xs = tf.tensor2d([inputMapping['cat'], inputMapping['dog'], inputMapping['bird']], [3, 1]);

// Utilise l'encodage one-hot pour les étiquettes 'chat', 'chien', 'oiseau'
const ys = tf.tensor2d([
  [1, 0, 0], // chat
  [0, 1, 0], // chien
  [0, 0, 1], // oiseau
], [3, 3]);

// Entraîne le modèle avec les données (xs, ys) pendant 1000 époques
model.fit(xs, ys, { epochs: 1000 }).then(() => {
  // Prédit la sortie pour une nouvelle entrée (cat, dog ou bird) et affiche le résultat
  const inputString = 'cat';

  const prediction = model.predict(tf.tensor2d([inputMapping[inputString]], [1, 1]));
  prediction.print();

  // Affiche la prédiction en pourcentage
  const labels = ['chat', 'chien', 'oiseau'];
  prediction.data().then(predictionData => {
    const highestValueIndex = predictionData.indexOf(Math.max(...predictionData));
    console.log(`Prédiction : ${labels[highestValueIndex]} avec ${(predictionData[highestValueIndex] * 100).toFixed(2)}% de confiance.`);
  });
});