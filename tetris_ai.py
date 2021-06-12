import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from game import Game
from tetromino import Tetromino
import random
import pickle
from common import *
from gui import Gui
import time
import multiprocessing

# size dependent
shape_main_grid = (1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1)
if STATE_INPUT == 'short':
    shape_hold_next = (1, 1 * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = 1 * 2 + 1
else:
    shape_hold_next = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = GAME_BOARD_WIDTH * 2 + 1

gamma = 0.95
epsilon = 0.06

current_avg_score = 0
rand = random.Random()

penalty = -500
reward_coef = [1.0, 0.5, 0.4, 0.3]


# reward_coef = [1.0, 1.0, 1.0, 1.0]


def make_model_conv2d():
    main_grid_input = keras.Input(shape=shape_main_grid[1:], name="main_grid_input")
    a = layers.Conv2D(
        64, 6, activation="relu", input_shape=shape_main_grid[1:]
    )(main_grid_input)
    a1 = layers.MaxPool2D(pool_size=(15, 5), strides=(1, 1))(a)
    a1 = layers.Flatten()(a1)
    a2 = layers.AvgPool2D(pool_size=(15, 5))(a)
    a2 = layers.Flatten()(a2)

    b = layers.Conv2D(
        256, 4, activation="relu", input_shape=shape_main_grid[1:]
    )(main_grid_input)
    b1 = layers.MaxPool2D(pool_size=(17, 7), strides=(1, 1))(b)
    b1 = layers.Flatten()(b1)
    b2 = layers.AvgPool2D(pool_size=(17, 7))(b)
    b2 = layers.Flatten()(b2)

    hold_next_input = keras.Input(shape=shape_hold_next[1:], name="hold_next_input")

    x = layers.concatenate([a1, a2, b1, b2, hold_next_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    critic_output = layers.Dense(1)(x)  # activation=None -> 'linear'

    model_new = keras.Model(
        inputs=[main_grid_input, hold_next_input],
        outputs=critic_output
    )

    model_new.summary()

    return model_new


def load_model(filepath=None):
    model_loaded = make_model_conv2d()

    model_loaded.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='huber_loss',
        metrics='mean_squared_error'
    )
    if filepath is not None:
        model_loaded.load_weights(filepath)
    else:
        model_loaded.save(FOLDER_NAME + 'whole_model/outer_{}'.format(0))
        print('model initial state has been saved')

    return model_loaded


def test(model, max_games=100, mode='piece', is_gui_on=True):
    max_steps_per_episode = 2000
    seed = None
    gui = Gui() if is_gui_on else None
    # env = GameMini(seed=seed, height=0)
    env = Game(gui=gui, seed=seed, height=0)

    episode_count = 0
    total_score = 0

    pause_time = 0.00

    while True and episode_count < max_games:
        env.reset()
        for step in range(max_steps_per_episode):
            states, add_scores, dones, _, _, moves = env.get_all_possible_states_conv2d()
            rewards = get_reward(add_scores, dones)
            q = rewards + model(states)
            best = tf.argmax(q).numpy()[0]

            if mode == 'step':
                best_moves = moves[best]

                for i in range(len(best_moves) - 1):
                    move = best_moves[i]
                    env.step(action=move)
                    env.render()
                    time.sleep(pause_time)
                env.step(chosen=best)
                env.render()
                time.sleep(pause_time)
            else:
                env.step(chosen=best)
                env.render()
                time.sleep(pause_time)

            if env.is_done() or step == max_steps_per_episode - 1:
                episode_count += 1
                total_score += env.current_state.score
                print('episode #{}:   score:{}'.format(episode_count, env.current_state.score))
                break

    print('average score = {:7.2f}'.format(total_score / max_games))


def get_data_from_playing_cnn2d(model_filename, target_size=8000, max_steps_per_episode=2000, proc_num=0,
                                queue=None):
    tf.autograph.set_verbosity(3)
    model = keras.models.load_model(model_filename)
    if model is None:
        print('ERROR: model has not been loaded. Check this part.')
        exit()

    global epsilon
    if proc_num == 0:
        epsilon = 0

    data = list()
    env = Game()
    episode_max = 1000
    total_score = 0
    avg_score = 0
    t_spins = 0

    for episode in range(episode_max):
        # env.reset(rand.randint(0, 10))
        env.reset()
        episode_data = list()
        for step in range(max_steps_per_episode):
            s = env.get_state_dqn_conv2d(env.current_state)
            possible_states, add_scores, dones, is_include_hold, is_new_hold, _ = env.get_all_possible_states_conv2d()
            rewards = get_reward(add_scores, dones)

            pool_size = Tetromino.pool_size()

            # get the best first before modifying the last next
            q = rewards + model(possible_states, training=False).numpy()
            for j in range(len(dones)):
                if dones[j]:
                    q[j] = rewards[j]
            best = tf.argmax(q).numpy()[0] + 0

            # if hold was empty, then we don't know what's next; if hold was not empty, then we know what's next!
            if is_include_hold and not is_new_hold:
                possible_states[1][:-1, -pool_size:] = 0
            else:
                possible_states[1][:, -pool_size:] = 0

            rand_fl = rand.random()
            if rand_fl > epsilon:
                chosen = best
            else:
                # probability based on q
                # q_normal = q.reshape(-1)
                # q_normal = q_normal - np.min(q_normal) + 0.001
                # q_normal = q_normal / np.sum(q_normal) + 0.3
                # q_normal = q_normal / np.sum(q_normal)
                # chosen = np.random.choice(q_normal.shape[0], p=q_normal)

                # uniform probability
                chosen = random.randint(0, len(dones) - 1)

            episode_data.append(
                (s, (possible_states[0][best], possible_states[1][best]), add_scores[best], dones[best]))

            if add_scores[best] != int(add_scores[best]):
                t_spins += 1

            env.step(chosen=chosen)

            if env.is_done() or step == max_steps_per_episode - 1:
                data += episode_data
                total_score += env.current_state.score
                break

        if len(data) > target_size:
            print('proc_num: #{:<2d} | total episodes:{:<4d} | avg score:{:<7.2f} | data size:{} | t-spins: {}'.format(
                proc_num, episode + 1, total_score / (episode + 1), len(data), t_spins))
            avg_score = total_score / (episode + 1)
            break

    if queue is not None:
        queue.put((data, avg_score), block=False)
        return

    return data, avg_score


def train(model, outer_start=0, outer_max=100):
    # outer_max: update samples
    inner_max = 5  # update target
    epoch_training = 5  # model fitting times
    batch_training = 512

    buffer_new_size = 20000
    buffer_outer_max = 1
    history = None

    for outer in range(outer_start + 1, outer_start + 1 + outer_max):
        print('======== outer = {} ========'.format(outer))
        time_outer_begin = time.time()

        # 1. collecting data.
        buffer = list()

        # getting new samples
        new_buffer = collect_samples_multiprocess_queue(model_filename=FOLDER_NAME + f'whole_model/outer_{outer - 1}',
                                                        outer=outer - 1, target_size=buffer_new_size)
        save_buffer_to_file(FOLDER_NAME + f'dataset/buffer_{outer}.pkl', new_buffer)
        buffer += new_buffer

        # load more samples
        for i in range(max(1, outer - buffer_outer_max + 1), outer):
            buffer += load_buffer_from_file(filename=FOLDER_NAME + 'dataset/buffer_{}.pkl'.format(i))

        random.shuffle(buffer)

        # 2. calculating target
        s1, s2, s1_, s2_, r_, dones_ = process_buffer_best(buffer)

        buffer_size = len(buffer)
        new_buffer_size = len(new_buffer)
        del buffer
        del new_buffer

        for inner in range(inner_max):
            print(f"      ======== inner = {inner + 1}/{inner_max} =========")
            target = list()
            for i in range(int(s1.shape[0] / batch_training) + 1):
                start = i * batch_training
                end = min((i + 1) * batch_training, s1.shape[0] + 1)
                target.append(
                    model((s1_[start:end, :, :, :], s2_[start:end, :]), training=False).numpy().reshape(-1) + r_[
                                                                                                              start:end])
            target = np.concatenate(target)
            # when it's gameover, Q[s'] must not be added
            for i in range(len(dones_)):
                if dones_[i]:
                    target[i] = r_[i]

            target = target * gamma

            history = model.fit((s1, s2), target, batch_size=batch_training, epochs=epoch_training, verbose=1)
            print('      loss = {:8.3f}   mse = {:8.3f}'.format(history.history['loss'][-1],
                                                                history.history['mean_squared_error'][-1]))

        model.save(FOLDER_NAME + 'whole_model/outer_{}'.format(outer))
        model.save_weights(FOLDER_NAME + 'checkpoints_dqn/outer_{}'.format(outer))

        time_outer_end = time.time()
        text_ = 'outer = {:>4d} | pre-training avg score = {:>8.3f} | loss = {:>8.3f} | mse = {:>8.3f} |' \
                ' dataset size = {:>7d} | new dataset size = {:>7d} | time elapsed: {:>6.1f} sec | coef = {} | penalty = {:>7d} | gamma = {:>6.3f}\n' \
            .format(outer, current_avg_score, history.history['loss'][-1], history.history['mean_squared_error'][-1],
                    buffer_size, new_buffer_size, time_outer_end - time_outer_begin, reward_coef, penalty, gamma
                    )
        append_record(text_)
        print('   ' + text_)


def save_buffer_to_file(filename, buffer):
    from pathlib import Path
    Path(FOLDER_NAME + 'dataset').mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)


def load_buffer_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def process_buffer_best(buffer):
    s1 = list()
    s2 = list()
    s1_ = list()
    s2_ = list()
    add_scores = list()
    dones_ = list()
    for row in buffer:
        s1.append(row[0][0])
        s2.append(row[0][1])
        s1_.append(row[1][0])
        s2_.append(row[1][1])
        add_scores.append(row[2])
        dones_ += [row[3]]

    s1 = np.concatenate(s1)
    s2 = np.concatenate(s2)
    s1_ = np.concatenate(s1_).reshape(s1.shape)
    s2_ = np.concatenate(s2_).reshape(s2.shape)
    r_ = get_reward(add_scores, dones_)
    r_ = np.concatenate(r_)
    return s1, s2, s1_, s2_, r_, dones_


def check_same_state(s1, s2):
    s1_ = s1.reshape(-1)
    s2_ = s2.reshape(-1)
    for i in range(s1_.shape[0]):
        if s1_[i] != s2_[i]: return False

    return True


def append_record(text, filename=None):
    if filename is None:
        filename = FOLDER_NAME + 'record.txt'
    with open(filename, 'a') as f:
        f.write(text)


def collect_samples_multiprocess_queue(model_filename, outer=0, target_size=10000):
    timeout = 800
    cpu_count = min(multiprocessing.cpu_count(), CPU_MAX)
    jobs = list()
    q = multiprocessing.Queue()
    for i in range(cpu_count):
        p = multiprocessing.Process(target=get_data_from_playing_cnn2d,
                                    args=(
                                        model_filename, int(target_size / cpu_count), 1000, i, q))
        jobs.append(p)
        p.start()

    data = list()
    scores = list()

    for i in range(cpu_count):
        d_, s_ = q.get(timeout=timeout)
        data += d_
        scores.append(s_)

    i = 0
    for proc in jobs:
        proc.join()
        i += 1

    # average score is max(scores) because it's the process with eps = 0
    print(f'end multiprocess: total data length: {len(data)} | avg score: {max(scores)}')
    global current_avg_score
    current_avg_score = max(scores)

    return data


def get_reward(add_scores, dones):
    reward = list()
    # manipulate the reward
    for i in range(len(add_scores)):
        add_score = add_scores[i].item()

        # give extra reward to t-spin
        # if add_score != int(add_score):
        #     add_score = add_score * 10
        #
        # if add_score >= 90:
        #     add_score = add_score * reward_coef[0]
        # elif add_score >= 50:
        #     add_score = add_score * reward_coef[1]
        # elif add_score >= 20:
        #     add_score = add_score * reward_coef[2]
        # elif add_score >= 5:
        #     add_score = add_score * reward_coef[3]

        if dones[i]:
            add_score += penalty
        reward.append(add_score)
    return np.array(reward).reshape([-1, 1])


if __name__ == "__main__":
    if MODE == 'human_player':
        game = Game(gui=Gui(), seed=None)
        game.restart()
        game.run()
    elif MODE == 'ai_player_training':
        if OUT_START == 0:
            load_model()
        model_load = keras.models.load_model(FOLDER_NAME + 'whole_model/outer_{}'.format(OUT_START))
        train(model_load, outer_start=OUT_START, outer_max=OUTER_MAX)
    elif MODE == 'ai_player_watching':
        model_load = keras.models.load_model(FOLDER_NAME + 'whole_model/outer_{}'.format(OUT_START))
        test(model_load, mode='step', is_gui_on=True)
