# LNR_project

The aim of the DETOXIS task is the detection of toxicity in comments posted in Spanish in response to different online news articles related to immigration. The DETOXIS task is divided into two related classification subtasks: Toxicity detection task and Toxicity level detection task.

The presence of toxic messages on social media and the need to identify and mitigate them leads to the development of systems for their automatic detection. The automatic detection of toxic language, especially in tweets and comments, is a task that has attracted growing interest from the NLP community in recent years.

The main novelty of the present task is, on the one hand, the methodology applied to the annotation of the dataset that will be used for training and testing the participant models and, on the other hand, the evaluation metrics that will be applied to evaluating the participant models in terms of their system use profile applying four different metrics (F-measure, Rank Biased Precision (Moffat et al. 2008), Closeness Evaluation Measure (Amigó et al., 2020) and Pearson’s correlation coefficient). The methodology proposed aims to reduce the subjectivity of the annotation of toxicity by taking into account the contextual information, i.e. the conversational thread, and by annotating different linguistic features, such as argumentation, constructiveness, stance, target, stereotype, sarcasm, mockery, insult, improper language, aggressiveness and intolerance, which allowed us to discriminate the different levels of toxicity. All this information will be included only in the training dataset that will be used for the task.

This following tasks/parts will be done by using Jupyter Notebooks:

- **Part 1**: Preprocessing and feature extraction
- **Part 2**: Classification
- **Part 3**: Evaluation
   - Metrics
   - Cross-validation
   - Validation curves
   - Setting parameters
- **Part 4**: Ensemble
   - Averaging
   - Boosting
- **Part 5**: Deep Learning
   - Transformers: BERT
