from math import fabs
import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')
learningRate = 0.01
discountRate = 0.99
nb_entry = env.observation_space.shape[0]
nb_hidden = 4
nb_action = 2
nb_train = 500
nb_try_per_train = 10
nb_max_step = 500
render_step = 50

def getGradientPlaceholders(grad_and_vars):
    gradient_placeholders = [] # Array of placeholders for gradients
    grad_and_vars_feed = [] # Array that will feed the optimizer with computed gradients
    for grad, var in grad_and_vars:
        tmp_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(tmp_placeholder)
        grad_and_vars_feed.append((tmp_placeholder, var))
    return gradient_placeholders, grad_and_vars_feed

# Return expected outputs of the neural network in function of the decision
# Used for calculate gradients
def getExpectedOut(action):
    out = []
    for i in range(nb_action):
            out.append(1. if i == action else 0.)
    return out

# Feed dictionary to give to the optimizer
def getGradientFeed(gradient_placeholders, gradients, all_rewards):
    feed_dict = {}
    for grad_idx, grad_placeholder in enumerate(gradient_placeholders):
        tmp_grad = [] # Gradient * Reward
        for try_index, rewards in enumerate(all_rewards): # For each try
            for step, reward in enumerate(rewards): # For each step in current try
                tmp_grad.append(reward * gradients[try_index][step][grad_idx]) # Gradient * reward
        mean_gradients = np.mean(tmp_grad, axis=0) # Mean gradients
        feed_dict[grad_placeholder] = mean_gradients
    return feed_dict

# Execute the discount operation on rewards
def discountRewards(rewards, discount_rate):
    out = []
    cumul = 0
    for i in reversed(range(len(rewards))):
        cumul = rewards[i] + (cumul * discount_rate)
        out.insert(0, cumul)
    return out

# Discount all rewards and normalize them
def discountAndNormalizeRewards(all_rewards, discount_rate):
    # Discount for each try
    discounted_rewards = [discountRewards(rewards, discount_rate) for rewards in all_rewards]
    flat = np.concatenate(discounted_rewards) # Concatenate all discounted rewards in a single list
    mean = flat.mean() # Mean
    std = flat.std() # Standard deviation
    return [(rewards - mean) / std for rewards in discounted_rewards] # Return normalized discounted rewards

# Execute tries to accumulate rewards and gradients
def executeTries(sess, entry, y, action, gradients, render):
    all_rewards = [] # Rewards of all tries in the train iteration
    all_gradients = [] # Gradients of all tries in the train iteration

    for try_number in range(nb_try_per_train):
        # Run a try
        obs = env.reset() # Reset environment & get first observation

        current_rewards = [] # Rewards of current try
        current_gradients = [] # Gradients of current try

        for i in range(nb_max_step):
            if render:
                env.render()

            a = action.eval(feed_dict={entry: [obs]})[0][0] # Take a decision
            expectedOuts = getExpectedOut(a) # Calculate the expected output of the neural network
            grads = sess.run(gradients, feed_dict={entry: [obs], y: [expectedOuts]}) # Get gradients

            obs, reward, done , info = env.step(a) # Execute the action in the environment
            reward -= fabs(obs[0]) / 2. # Substract the distance from the center (force it to be centered)
            current_rewards.append(reward)
            current_gradients.append(grads)

            if done: # Break if the try is finished
                break;
        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)

    return discountAndNormalizeRewards(all_rewards, discountRate), all_gradients

def main():
    # Construct neural network
    entry = tf.placeholder(tf.float32, shape=[None, nb_entry], name="entry")
    layer1 = tf.layers.dense(entry, nb_hidden, name="hidden", activation=tf.nn.selu)
    out=tf.layers.dense(layer1, nb_action, name="hidden2")

    # Action taken randomly with softmax outputs probabilities
    action = tf.multinomial(tf.log(tf.nn.softmax(out)), num_samples=1)

    y = tf.placeholder(tf.float32, shape=[None, nb_action], name="expectedOuts") # Expected activations
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y) # Loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    grad_and_vars = optimizer.compute_gradients(xentropy) # Compute gradients
    gradients = [grad for grad, var in grad_and_vars] # Get gradients

    # Get gradient placeholders to feed the optimizer
    gradient_placeholders, grad_and_vars_feed = getGradientPlaceholders(grad_and_vars)

    training_op = optimizer.apply_gradients(grad_and_vars_feed)

    init = tf.global_variables_initializer() # Variable initializer

    # Run the session
    with tf.Session() as sess:
        sess.run(init) # Initialize variables

        # Run the learning process
        for train_step in range(nb_train):
            print("Iteration:", train_step)
            # Execute tries and collect normalized discounted rewards & gradients
            all_rewards, all_gradients = executeTries(sess, entry, y, action, gradients,\
                                                      train_step % render_step == 0)
            # Compute policy gradients with rewards
            feed_dict = getGradientFeed(gradient_placeholders, all_gradients, all_rewards)
            # Execute policy gradient descent
            sess.run(training_op, feed_dict=feed_dict)

if __name__ == "__main__":
    main()
