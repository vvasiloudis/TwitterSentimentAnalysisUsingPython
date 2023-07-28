# TwitterSentimentAnalysisUsingPython

First, we took a dataset that includes users from
Tweeters who have posted texts about an airline
company of America and the main objective is the perception of the positive,
neutral and negative comments. At the beginning of the program you will
declare all the necessary libraries we will need
for the implementation of our answer and then we also import it
dataset we got. Then, with the shape command we see the columns
and all the elements that the dataset we have, Axis=0 says
basically in everything that we will drop the rows, whenever we find a column
with < 0.5. The shape of the data, when you print it, will appear
as (14404, 15), essentially means there are 14404 tweets in the
dataset, with 15 parameters (such as text, location, etc.)
associated with each tweet and then we will assign the columns that
we are interested in variables. Next, we enter all the
necessary libraries that will help us in processing
data and we will store all the English words stop and the
stopper function in different variables. Then I will
create an empty list called cleaned_data, where
the text data will be saved after we get rid of all of them
unnecessary words. We import a library called
re(regular expressions), which will help us remove a lot
unnecessary things. First, we execute a for loop to iterate over
every tweet in the dataset, every time. Using it
re.sub we can replace things in a
proposal. So basically we take one tweet at a time and within that
anything that is not a letter (belongs to az or AZ), will be replaced by one
empty space. It will automatically filter punctuation and other non-punctuation
letters and then we will convert all words to lowercase
letters and divide them into a list. After that we join all of them
words using ''.join(tweet), to get a sentence
instead of separated words. And then we just attach it
in the clean data list we had created. The function
Count Vectorizer converts a list of words into a bag of words and into
then we set max columns to 3000 and basically keep it
maximum displayed at 3000 words. Also, we defined her
stop_words parameter to the name of the airlines, as well
we want to remove her from the tweets as well. Finally, the function
cv.fit_transform takes the cleaned_data and transforms it into a bag of
words we wanted. Now, as for the input, we're going to convert it
output to numbers, this action will result in
replace the emotions "negative, neutral, positive" in
"0,1,2" . For each word in the Variable_Y column, we pass it through one
function called “sentiment_ordering.index()” which
it basically replaces the word with its index in the list
sentiment_ordering. Next, we will use the model
Multinomial Naive Bayes for the relationship between input and output.
Multinomial NB is a supervised learning algorithm
which works very well for text-based data.
Now to create the model we divide the whole
data in a training and test section (size
test=30% of real data). To suit
actually in the model, we call the function model.fit and its
we provide input and output training. Finally, to
check how good our model is, we first make predictions
for the test set and then print the report
of its classification which will tell us many important parameters such as
precision, recall, f1 score.
