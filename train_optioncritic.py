from general_train import init_arg_parse, get_model

parser = init_arg_parse()
# custom model arguments

args = parser.parse_args()
env, model, folder, callbacks, device, num_steps, logging_freq = get_model('OptionCritic', args)
model = model(env, update_frequency=1, logdir=folder, num_options=8, device=device)

# trainig
model.learn(total_timesteps=num_steps, log_interval=logging_freq, callback=callbacks)
model.save(folder + type(model).__name__ + "/final_model")
