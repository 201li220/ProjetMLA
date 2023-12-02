Stockage des modeles fonctionnel mais pb pour rechargement, meme erreur que sur le notebook de nico 

Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[51], line 2
      1 reload = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
----> 2 reload.load_state_dict(torch.load('Saved_Models/Nicoloco.pth'))
