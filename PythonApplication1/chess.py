import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

db = open("D:\\Users\\Damien\\Downloads\\chess\\db.txt", "r")
conv = [" ","r","n","b","k","q","p","R","N","B","K","Q","P"]

data=[]
for line in db:
    data.append(line)
db.close()

def uciToInt(uci):
    return (ord(uci[0])-ord('a'))

def intToUci(i):
    s = str(i)
    return chr(int(s[0])+ord('a'))

def fenToFloat(fen):
    board=np.empty([8,8], dtype=int)
    i=0
    j=0
    k=0
    while fen[k]!=" ":
        if fen[k]=="/":
            i+=1
            j=0
            k+=1
            continue
        elif fen[k] in conv:
            board[i][j] = conv.index(fen[k])/13
        elif fen[k].isdigit():
            for l in range(int(fen[k])):
                board[i][j]=0
                j+=1
            j-=1
        k+=1
        j+=1
    return board

x=[]
y=[]

for i in range(len(data)):
    if i%2==0:
        y.append(uciToInt(data[i]))
    else:
        x.append(fenToFloat(data[i]))

x_train=np.array(x[10000:], dtype=float)
y_train=np.array(y[10000:], dtype=int)
x_test=np.array(x[:10000], dtype=float)
y_test=np.array(y[:10000], dtype=int)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  color = 'blue' if predicted_label == true_label else 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(intToUci(predicted_label),
                                100*np.max(predictions_array),
                                intToUci(true_label)),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(8,8)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


print(model.evaluate(x_test, y_test))

predictions = model.predict(x_test)

num_rows = 10
num_cols = 10
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, y_test)
plt.show()













pgn = open("D:\\Users\\Damien\\Downloads\\chess\\lichess_db_standard_rated_2018-12.pgn", "r")
db = open("D:\\Users\\Damien\\Downloads\\chess\\db.txt", "w")
data=[]

game = chess.pgn.read_game(pgn)
while game != None:
    board = game.end().board()
    if board.is_checkmate() and not board.turn:
        db.write(board.pop().uci())
        db.write("\n")
        db.write(board.fen())
        db.write("\n")
    game = chess.pgn.read_game(pgn)

db.close()