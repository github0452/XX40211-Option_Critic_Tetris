from general_train import init_arg_parse, get_model

parser = init_arg_parse()
# custom model arguments

args = parser.parse_args()
env, model, folder, callbacks, device, num_steps, logging_freq = get_model('DQN', args)
model = model("CnnPolicy", env, verbose=0, buffer_size=10000, tensorboard_log=folder, device=device)

# trainig
model.learn(total_timesteps=num_steps, log_interval=logging_freq, callback=callbacks)
model.save(folder + type(model).__name__ + "/final_model")
