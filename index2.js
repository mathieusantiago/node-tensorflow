const tf = require('@tensorflow/tfjs');
const nlp = require('@tensorflow-models/universal-sentence-encoder');

// Charge le modèle pré-entraîné
nlp.load().then(model => {
  // Encode des phrases en vecteurs
  const phrases = ['Le chat est sur le canapé.', 'Le chien dort dans son panier.'];
  model.embed(phrases).then(embeddings => {
    // Les embeddings sont des vecteurs de taille 512 pour chaque phrase
    embeddings.print();

    // Vous pouvez maintenant utiliser ces embeddings pour diverses tâches de NLP, telles que la classification, la similarité de texte, etc.
  });
});