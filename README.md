# Text generator with Recurrent Neural Networks (RNN)

## Introduction

In this activity we will load a corpus or dataset containing a lot of text, and we will train a LSTM to predict characters in order to generate text from a seed. The idea is that this generated text froms coherent sentences that are dependent on the context of the dataset.
As a dataset, we will use an extract of the book El Quijote from Miguel de Cervantes (text in Spanish language), but you can load your own text dataset.

The LSTM have 99 hidden states of dimension 128, and receives an input of length *n=99*, and outputs a single value at the end. The inputs and outputs are characters represented with one-hot encoding. The size of the one-hot encoding is the size of the vocabulary of the dataset (filtering some characters like signs and punctuations)

The idea for training is passing *n* characters of the text as input, and the target output value would be the next character that follows the last character in the input. For example, if *n=10* and the dataset text is `Hello, my name is John and..`, we could have:

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;x_1&space;=&space;\texttt{'Hello,&space;my&space;'}&space;\hspace{50}&space;y_1&space;=&space;\texttt{'n'}&space;\\&space;x_2&space;=&space;\texttt{'ello,&space;my&space;n'}&space;\hspace{50}&space;y_2&space;=&space;\texttt{'a'}&space;\\&space;x_3&space;=&space;\texttt{'llo,&space;my&space;na'}&space;\hspace{50}&space;y_3&space;=&space;\texttt{'m'}&space;\\&space;x_4&space;=&space;\texttt{'lo,&space;my&space;nam'}&space;\hspace{50}&space;y_4&space;=&space;\texttt{'e'}&space;\\&space;x_5&space;=&space;\texttt{'o,&space;my&space;name'}&space;\hspace{50}&space;y_6&space;=&space;\texttt{'&space;'}&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;x_1&space;=&space;\texttt{'Hello,&space;my&space;'}&space;\hspace{50}&space;y_1&space;=&space;\texttt{'n'}&space;\\&space;x_2&space;=&space;\texttt{'ello,&space;my&space;n'}&space;\hspace{50}&space;y_2&space;=&space;\texttt{'a'}&space;\\&space;x_3&space;=&space;\texttt{'llo,&space;my&space;na'}&space;\hspace{50}&space;y_3&space;=&space;\texttt{'m'}&space;\\&space;x_4&space;=&space;\texttt{'lo,&space;my&space;nam'}&space;\hspace{50}&space;y_4&space;=&space;\texttt{'e'}&space;\\&space;x_5&space;=&space;\texttt{'o,&space;my&space;name'}&space;\hspace{50}&space;y_6&space;=&space;\texttt{'&space;'}&space;\\" title="\\ x_1 = \texttt{'Hello, my '} \hspace{50} y_1 = \texttt{'n'} \\ x_2 = \texttt{'ello, my n'} \hspace{50} y_2 = \texttt{'a'} \\ x_3 = \texttt{'llo, my na'} \hspace{50} y_3 = \texttt{'m'} \\ x_4 = \texttt{'lo, my nam'} \hspace{50} y_4 = \texttt{'e'} \\ x_5 = \texttt{'o, my name'} \hspace{50} y_6 = \texttt{' '} \\" /></a>

(Note that we have a stride of 1 for the training samples. Also here we use *n=10* as an example, but we really use *n=99*)

The whole text will be used for training, and there is no validation or test set, because we only want to generate text from a text seed of the corpus.

## Requirements

- Python 3.x
- Tensorflow
- Keras

## Model and training

The configuration of the model is:

* Model: LSTM
* Input length: 99
* Output length: 1
* Hidden states dimension: 128 
* Dropout of 0.2
* Softmax activation at the end (because of one-hot configuration)

The parameters for training are:

* Optimizer: Adam
* Learning rate: 0.0005
* Batch size: 128
* Epochs: 60

## Results

In this section we show the results of the text generation starting form a seed, which corresponds to a text of 99 characters, and then, from there the generator begins to invent text up to an extension of 300 more characters.

We use different temperatures for sampling the character out of the output probability vector, instead of simply use the argmax to match de one-hot encoded character. This allow more diversity in the results. (At low temperatures, more probability to choose the argmax).

In the first epochs we can see that words and sentences are not very accurate and also there is a lot of repetition.

```
Epoch 5/60

loss: 1.6648 - acc: 0.4776

Seed generator: "suelo. un mozo que iba a pie, viendo caer al encamisado, comenzo a denostar a don quijote, el cual,"

----- temperature: 0.2
suelo. un mozo que iba a pie, viendo caer al encamisado, comenzo a denostar a don quijote, el cual, que esta en la vertado a la caballero y de la mencida, y al cual de la mana de su estaba en el caballero a la caballero, y al como de los caballeros de su habia de la caballero a deste de los despondio el caballero de su amo a su serredo el caballero, y al caballero a deste su habia de su caballero

----- temperature: 0.5
suelo. un mozo que iba a pie, viendo caer al encamisado, comenzo a denostar a don quijote, el cual, halia de poraba su entendir en esto estaba de su hacia para las caballeros de la que de alguna buena alguna hacer en el cual de la cual de lo que los condes con la mas andan de la suber de los dios dio don quijote, sin dio en el valio por le hacia como le perdoba de todos los que se la sencido por 

----- temperature: 1.0
suelo. un mozo que iba a pie, viendo caer al encamisado, comenzo a denostar a don quijote, el cual, de voy el donte dio mas espuestrie un famose que esficio suerias poros hen algo, por que haman musha de las mas mas delvicio que podrjano que yenguras fues. quera modadar suncido anfentio casa, se puerve sen vies. asio, se hayeivodo oterende las sobrecos que fueso de queres hacia festendo en las qu

----- temperature: 1.2
suelo. un mozo que iba a pie, viendo caer al encamisado, comenzo a denostar a don quijote, el cual, la que entelpeuel manza, me dino may ariana, no elara de oroas, coso lio, que el riemo en el victo dllgo que no her ago? señor yabo zas decer a harmera, y recerte la mañas ficoti, le caia dorvio  u docer que allo quisa deban y das simo, que la dije: mis de muy destos yo ol cor armentarn. yo hi ne s
```

After several epochs, we can begin to notice more coherence and accuracy in the structure of words and sentences. Also at higher temperatures we can see less repetition, but if the temperature is too high, we still get incoherent words

```
Epoch 42/60

loss: 1.1773 - acc: 0.6199

Seed generator: "si  no saben que decirle sino llamarla a voces cruel y desagradecida,  con otros titulos a este sem"

----- temperature: 0.2
si  no saben que decirle sino llamarla a voces cruel y desagradecida,  con otros titulos a este semejante caballero       en esto digo, señor mio, respondio el caballo, y asi como el paso en el cuello del camino del camino de la caballeria andante, y asi como el pensaba que estaba en la mano y todos los caballeros andantes, y es verdad que estaba en la cabeza y en casa de la mano y la venta, y ll

----- temperature: 0.5
si  no saben que decirle sino llamarla a voces cruel y desagradecida,  con otros titulos a este semejante caballero andante.       comiendole de nuestra casa que le habian pasado o alguna gran parte de su caballo, y sin que era el pobre dicho, sin que era la venta que la sin par y caballeria, a rocinante, y llegandose con una pensas de alli adelante que no pudieran tiempo que eran conteniose y es

----- temperature: 1.0
si  no saben que decirle sino llamarla a voces cruel y desagradecida,  con otros titulos a este semejante sin nos, sin saber. los que parecia sen le compusta. asi es, que a los armasiles de que acudieno sen, dijo el cura, que era en la cabeza; el cual diciendo: entra azvueltas o quite les todes es verdadero que tenian los que poner sobre su señora tuvo, viendo don quijote: y este por liciticas, q

----- temperature: 1.2
si  no saben que decirle sino llamarla a voces cruel y desagradecida,  con otros titulos a este semejante con el  don ialcas; mas digan dias di ver el camino y suntancia no seraimene. uno durosano: entro ordas, llegaron tu causido caballero, sino a sosisar a dejar que le hagien del mundo; porque repues. señor, sambearme, para: marden es, nunca agravios se estibo con alcanzara y tienca; porque esc
```
