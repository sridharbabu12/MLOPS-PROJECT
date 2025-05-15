from scipy.stats import randint,uniform

LIGHTGM_PARAMS = {
    'n_estimators': randint(100,500),
    'max_depth':randint(5,50),
    'learning_rate':uniform(0.01,0.3),
    'num_leaves':randint(20,100),
    'boosting_type':['gbdt','dart','goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter':10,
    'cv':3,
    'n_jobs':-1,
    'verbose':1,
    'random_state':42,
    'scoring':'accuracy'
}
