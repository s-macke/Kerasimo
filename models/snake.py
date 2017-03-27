from keras.models import Sequential, load_model
from keras.layers import *
from qlearning4k.games import Snake
from keras.optimizers import *
from qlearning4k import Agent
from lib import kerasimo

grid_size = 10
nb_frames = 4
nb_actions = 5

snake = Snake(grid_size)

model = load_model('models/snake.hdf5')
#model = Sequential()
#model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(nb_frames, grid_size, grid_size)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dense(nb_actions))
#model.compile(RMSprop(), 'MSE')

agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
#model.save('/tmp/snake1.hdf5')
#agent.train(snake, batch_size=64, nb_epoch=10000, gamma=0.8)
#model.save('/tmp/snake2.hdf5')
#agent.play(snake)

snake.reset()
agent.clear_frames()
S = agent.get_game_data(snake)
game_over = False
frames = list()
frames.append(S[0])
while not game_over:
	q = model.predict(S)[0]
	possible_actions = snake.get_possible_actions()
	q = [q[i] for i in possible_actions]
	action = possible_actions[np.argmax(q)]
	snake.play(action)
	S = agent.get_game_data(snake)
	frames.append(S[0])
	game_over = snake.is_over()
print(np.asarray(frames).shape)
kerasimo.ToSVG('snake', model, np.array(frames), showarrows=False, columns=[1,3,3,10,10,1])
