"""The bias-variance tradeoff

    Often, researchers use the terms "bias" and "variance" or "bias-
    variance tradeoff" to describe the performance of a model—that
    is, you may stumble upon talks, books, or articles where people
    say that a model has a "high variance" or "high bias." So, what
    does that mean? In general, we might say that "high variance"
    is proportional to overfitting and "high bias" is proportional to
    underfitting.
    In the context of machine learning models, variance measures
    the consistency (or variability) of the model prediction for
    classifying a particular example if we retrain the model multiple
    times, for example, on different subsets of the training dataset.
    We can say that the model is sensitive to the randomness
    in the training data. In contrast, bias measures how far off
    the predictions are from the correct values in general if we
    rebuild the model multiple times on different training datasets;
    bias is the measure of the systematic error that is not due
    to randomness.

    Accurate definitions can be found in below link:

https://sebastianraschka.com/pdf/lecture-notes/stat479fs18/08_eval-intro_notes.pdf

"""
