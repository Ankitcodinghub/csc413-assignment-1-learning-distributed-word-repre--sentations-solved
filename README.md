# csc413-assignment-1-learning-distributed-word-repre--sentations-solved
**TO GET THIS SOLUTION VISIT:** [CSC413 Assignment 1-Learning Distributed Word Repre- sentations Solved](https://www.ankitcodinghub.com/product/csc413-assignment-1-learning-distributed-word-repre-sentations-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100107&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413 Assignment 1-Learning Distributed Word Repre- sentations Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
&nbsp;

Introduction

In this assignment we will learn about word embeddings and make neural networks learn about words. We could try to match statistics about the words, or we could train a network that takes a sequence of words as input and learns to predict the word that comes next.

This assignment will ask you to implement a linear embedding and then the backpropagation computations for a neural language model and then run some experiments to analyze the learned representation. The amount of code you have to write is very short but each line will require you to think very carefully. You will need to derive the updates mathematically, and then implement them using matrix and vector operations in NumPy.

Starter code and data

Download and extract the archive from the course web page https://csc413-2020.github.io/ assets/misc/a1-code.zip.

Look at the file raw_sentences.txt. It contains the sentences that we will be using for this assignment. These sentences are fairly simple ones and cover a vocabulary of only 250 words.

We have already extracted the 4-grams from this dataset and divided them into training, vali- dation, and test sets. To inspect this data, run the following within IPython:

1https://markus.teach.cs.toronto.edu/csc413-2020-01 2https://csc413-2020.github.io/assets/misc/syllabus.pdf

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
&nbsp;

<pre>            import pickle
            data = pickle.load(open(’data.pk’, ’rb’))
</pre>
Now data is a Python dict which contains the vocabulary, as well as the inputs and targets for all three splits of the data. data[’vocab’] is a list of the 250 words in the dictionary; data[’vocab’][0] is the word with index 0, and so on. data[’train_inputs’] is a 372, 500 × 3 matrix where each row gives the indices of the 3 context words for one of the 372,500 training cases. data[’train_targets’] is a vector giving the index of the target word for each training case. The validation and test sets are handled analogously.

Now look at the file language model.ipynb, which contains the starter code for the assignment. Even though you only have to modify a few specific locations in the code, you may want to read through this code before starting the assignment.

Part 1: Linear Embedding – GLoVE (4pts)

In this section we will be implementing a simplified version of GLoVE [Jeffrey Pennington and Manning]. Given a corpus with V distinct words, we define the co-occurrence matrix X ∈ V × V with entries Xij representing the frequency of the i-th word and j-th word in the corpus appearing in the same context – in our case the adjacent words. GLoVE aims to find a d-dimensional embedding of the words that preserves properties of the co-occurrence matrix by representing the i-th word with a d-dimensional vector wi ∈ Rd and a scalar bias bi ∈ R. This objective can be written as 3:

V

L({wi,bi}Vi=1)= 􏰆(wi⊤wj +bi +bj −logXij)2

i,j =1

When the bias terms are omitted, then GLoVE corresponds to finding a rank-d symmetric factor- ization of the co-occurrence matrix.

<ol>
<li>Given the vocabulary size V and embedding dimensionality d, how many trainable parameters does the GLoVE model have?</li>
<li>Write the gradient of the loss function with respect to one parameter vector wi.</li>
<li>Implement the gradient update of GLoVE in language model.ipynb.</li>
<li>Train the model with varying dimensionality d. Which d leads to optimal validation perfor- mance? Why does / doesn’t larger d always lead to better validation error?</li>
</ol>
3We have simplified the objective by omitting the weighting function and the additional weight vector w ̃, due to the symmetry of our co-occurrence matrix. For the complete algorithm please see [Jeffrey Pennington and Manning]

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
&nbsp;

Part 2: Network architectur

In this assignment, we will train a neural language model like the one we covered in lecture and as in Bengio et al. [2003]. It receives as input 3 consecutive words, and its aim is to predict a distribution over the next word (the target word). We train the model using the cross-entropy criterion, which is equivalent to maximizing the probability it assigns to the targets in the training set. Hopefully it will also learn to make sensible predictions for sequences it hasn’t seen before.

The model architecture is as follows:

The network consists of an input layer, embedding layer, hidden layer and output layer. The input consists of a sequence of 3 consecutive words, given as integer valued indices. (I.e., the 250 words in our dictionary are arbitrarily assigned integer values from 0 to 249.) The embedding layer maps each word to its corresponding vector representation. This layer has 3 × D units, where D is the embedding dimension, and it essentially functions as a lookup table. We share the same lookup table between all 3 positions, i.e. we don’t learn a separate word embedding for each context position. The embedding layer is connected to the hidden layer, which uses a logistic nonlinearity. The hidden layer in turn is connected to the output layer. The output layer is a softmax over the 250 words.

As a warm-up, please answer the following questions, each worth 1 point.

1. As above, assume we have 250 words in the dictionary and use the previous 3 words as inputs. Suppose we use a 16-dimensional word embedding and a hidden layer with 128 units. The

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
trainable parameters of the model consist of 3 weight matrices and 2 sets of biases. What is the total number of trainable parameters in the model? Which part of the model has the largest number of trainable parameters?

2. Another method for predicting the next word is an n-gram model, which was mentioned in Lecture 7. If we wanted to use an n-gram model with the same context length as our network, we’d need to store the counts of all possible 4-grams. If we stored all the counts explicitly, how many entries would this table have?

Part 3: Training the Neural Network (4pts)

In this part of the assignment, you implement a method which computes the gradient using back- propagation. To start you out, the Model class contains several important methods used in training:

• compute_activations computes the activations of all units on a given input batch

• compute_loss computes the total cross-entropy loss on a mini-batch

• evaluate computes the average cross-entropy loss for a given set of inputs and targets

You will need to complete the implementation of two additional methods which are needed for training:

• compute_loss_derivative computes the derivative of the loss function with respect to the output layer inputs. In other words, if C is the cost function, and the softmax computation

</div>
</div>
<div class="layoutArea">
<div class="column">
is

</div>
<div class="column">
ezi

yi = 􏰏j ezj ,

</div>
</div>
<div class="layoutArea">
<div class="column">
this function should compute a B × NV matrix where the entries correspond to the partial derivatives ∂C/∂zi.

• back_propagate is the function which computes the gradient of the loss with respect to model parameters using backpropagation. It uses the derivatives computed by compute_loss_derivative. Some parts are already filled in for you, but you need to compute the matrices of derivatives

for embed_to_hid_weights, hid_bias, hid_to_output_weights, and output_bias. These matrices have the same sizes as the parameter matrices (see previous section).

In order to implement backpropagation efficiently, you need to express the computations in terms of matrix operations, rather than for loops. You should first work through the derivatives on pencil and paper. First, apply the chain rule to compute the derivatives with respect to individual units, weights, and biases. Next, take the formulas you’ve derived, and express them in matrix form. You should be able to express all of the required computations using only matrix multiplication, matrix

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
&nbsp;

transpose, and elementwise operations — no for loops! If you want inspiration, read through the code for Model.compute_activations and try to understand how the matrix operations correspond to the computations performed by all the units in the network.

To make your life easier, we have provided the routine check_gradients, which checks your gradients using finite differences. You should make sure this check passes before continuing with the assignment.

Once you’ve implemented the gradient computation, you’ll need to train the model. The func- tion train in language model.ipynb implements the main training procedure. It takes two argu- ments:

• embedding_dim: The number of dimensions in the distributed representation.

• num_hid: The number of hidden units For example, execute the following:

<pre>                              model = train(16, 128)
</pre>
As the model trains, the script prints out some numbers that tell you how well the training is going. It shows:

<ul>
<li>The cross entropy on the last 100 mini-batches of the training set. This is shown after every 100 mini-batches.</li>
<li>The cross entropy on the entire validation set every 1000 mini-batches of training.
At the end of training, this function shows the cross entropies on the training, validation and test sets. It will return a Model instance.

To convince us that you have correctly implemented the gradient computations, please include the following with your assignment submission:
</li>
</ul>
<ul>
<li>You will submit language model.ipynb through MarkUs. You do not need to modify any of the code except the parts we asked you to implement.</li>
<li>In your writeup, include the output of the function print_gradients. This prints out part of the gradients for a partially trained network which we have provided, and we will check them against the correct outputs. Important: make sure to give the output of print_gradients, not check_gradients.
This is worth 4 points: 1 for the loss derivatives, 1 for the bias gradients, and 2 for the weight gradients. Since we gave you a gradient checker, you have no excuse for not getting full points on this part.
</li>
</ul>
</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
&nbsp;

Part 4: Analysis (4pts)

In this part, you will analyze the representation learned by the network. You should first train a model with a 16-dimensional embedding and 128 hidden units, as discussed in the previous section; you’ll use this trained model for the remainder of this section. Important: if you’ve made any fixes to your gradient code, you must reload the language_model module and then re-run the training procedure. Python does not reload modules automatically, and you don’t want to accidentally analyze an old version of your model.

These methods of the Model class can be used for analyzing the model after the training is done.

• tsne_plot creates a 2-dimensional embedding of the distributed representation space using an algorithm called t-SNE [Maaten and Hinton, 2008]. You don’t need to know what this is for the assignment, but we may cover it later in the course. Nearby points in this 2-D space are meant to correspond to nearby points in the 16-D space. From the learned model, you can create pictures that look like this:

• display_nearest_words lists the words whose embedding vectors are nearest to the given word

• word_distance computes the distance between the embeddings of two words 6

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
&nbsp;

• predict_next_word shows the possible next words the model considers most likely, along with their probabilities

Using these methods, please answer the following questions, each of which is worth 1 point.

<ol>
<li>Pickthreewordsfromthevocabularythatgowelltogether(forexample,‘government of united’, ‘city of new’, ‘life in the’, ‘he is the’ etc.). Use the model to predict the next word.

Does the model give sensible predictions? Try to find an example where it makes a plausible prediction even though the 4-gram wasn’t present in the dataset (raw_sentences.txt). To

help you out, the function find_occurrences lists the words that appear after a given 3-gram in the training set.
</li>
<li>Plot the 2-dimensional visualization using the method tsne_plot_representation. Look at

the plot and find a few clusters of related words. What do the words in each cluster have in common? Plot the 2-dimensional visualization using the method tsne_plot_GLoVE_representation for a 256 dimensional embedding. How do the t-SNE embeddings for both models compare?

Plot the 2-dimensional visualization using the method plot_2d_GLoVE_representation.

How does this compare to the t-SNE embeddings? (You don’t need to include the plots

with your submission.)</li>
<li>Are the words ‘new’ and ‘york’ close together in the learned representation? Why or why not?</li>
<li>Which pair of words is closer together in the learned representation: (‘government’, ‘political’), or (‘government’, ‘university’)? Why do you think this is?</li>
</ol>
What you have to submit

For reference, here is everything you need to hand in. See the top of this handout for submission directions.

• A PDF file titled a1-writeup.pdf containing the following:

– Answers to questions from Part 1

– Answers to questions from Part 2

– The output of print_gradients()

– Answers to all four questions from Part 4

• Your code file language model.ipynb

7

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
CSC413/2516 Winter 2020 with Professor Jimmy Ba Programming Assignment 1

References

Richard Socher Jeffrey Pennington and Christopher D Manning. Glove: Global vectors for word representation. Citeseer.

Yoshua Bengio, R ́ejean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. Journal of machine learning research, 3(Feb):1137–1155, 2003.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579–2605, 2008.

</div>
</div>
<div class="layoutArea">
<div class="column">
8

</div>
</div>
</div>
