
BASE_CONFIG = {
    'change_point_epoch': 10,
    'evaluation_interval': 1000,
    'num_epochs': 100,
    'delta_easy': 1e-3,
    'delta_hard': 1e-7,
    'seed0': 0,
    'seed1': 1000,
    'update_delta_dropped_accuracy': 1.0,
    'algo_vec_len': 2,
    'model': 'CutEFDTClassifier',
    'stream_type': 'RandomTree',
    'stream': {
            'n_classes': 3,
            'n_num_features': 3,
            'n_cat_features': 3,
            'n_categories_per_feature': 3,
            'max_tree_depth': 5,
            'first_leaf_level': 3,
            'fraction_leaves_per_level': 0.15,
    },
    'num_episodes': 100,
    'action_delay': 5,
    'debug': False,
    'alpha': 0.1,
    'gamma': 0.9,
    'epsilon': 0.5,
    'baseline_model': 'UpdatableEFDTClassifier',
}

STREAMS = [

    # {
    # 'stream_type': 'RandomTree',
    #  'stream' : {
    #     'n_classes': 2,
    #     'n_num_features': 2,
    #     'n_cat_features': 2,
    #     'n_categories_per_feature': 2,
    #     'max_tree_depth': 5,
    #     'first_leaf_level': 2,
    #     'fraction_leaves_per_level': 0.15,
    #     }
    # },


    {
    'stream_type': 'RandomTree',
     'stream' : {
        'n_classes': 3,
        'n_num_features': 3,
        'n_cat_features': 3,
        'n_categories_per_feature': 3,
        'max_tree_depth': 5,
        'first_leaf_level': 2,
        'fraction_leaves_per_level': 0.15,
        }
    },

    # {
    # 'stream_type': 'RandomTree',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_num_features': 4,
    #     'n_cat_features': 3,
    #     'n_categories_per_feature': 3,
    #     'max_tree_depth': 5,
    #     'first_leaf_level': 2,
    #     'fraction_leaves_per_level': 0.15,
    #     }
    # },

    # {
    # 'stream_type': 'RandomTree',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_num_features': 4,
    #     'n_cat_features': 4,
    #     'n_categories_per_feature': 3,
    #     'max_tree_depth': 5,
    #     'first_leaf_level': 2,
    #     'fraction_leaves_per_level': 0.15,
    #     }
    # },

    # {
    # 'stream_type': 'RandomTree',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_num_features': 4,
    #     'n_cat_features': 4,
    #     'n_categories_per_feature': 4,
    #     'max_tree_depth': 5,
    #     'first_leaf_level': 2,
    #     'fraction_leaves_per_level': 0.15,
    #     }
    # },


    # {
    # 'stream_type': 'RandomRBF',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_features': 5,
    #     'n_centroids': 10,
    #     }
    # },

    # {
    # 'stream_type': 'RandomRBF',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_features': 10,
    #     'n_centroids': 20,
    #     }
    # },


    # {
    # 'stream_type': 'RandomRBF',
    #  'stream' : {
    #     'n_classes': 2,
    #     'n_features': 10,
    #     'n_centroids': 50,
    #     }
    # },

    # {
    # 'stream_type': 'RandomRBF',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_features': 10,
    #     'n_centroids': 50,
    #     }
    # },

    # {
    # 'stream_type': 'RandomRBF',
    #  'stream' : {
    #     'n_classes': 4,
    #     'n_features': 20,
    #     'n_centroids': 20,
    #     }
    # },

    # {
    # 'stream_type': 'RandomTree',
    #  'stream' : {
    #     'n_classes': 5,
    #     'n_num_features': 5,
    #     'n_cat_features': 5,
    #     'n_categories_per_feature': 5,
    #     'max_tree_depth': 5,
    #     'first_leaf_level': 2,
    #     'fraction_leaves_per_level': 0.15,
    #     }
    # },

    # {
    # 'stream_type': 'Hyperplane',
    #  'stream' : {
    #      'n_features': 10,
    #      'noise_percentage': 0.05,
    #      'sigma': 0.1,
    #     }
    # },

    # {
    # 'stream_type': 'Hyperplane',
    #  'stream' : {
    #      'n_features': 20,
    #      'noise_percentage': 0.1,
    #      'sigma': 0.2,
    #     }
    # },

    # {
    # 'stream_type': 'SEA',
    #  'stream' : {
    #     # 'variant': 0, 
    #     # 'noise': 0.0
    #     # }
    # },




    # {
    # 'stream_type': 'Sine',
    #  'stream' : {
    #     'classification_function': 0,
    #     'balance_classes': False,
    #     'has_noise': True
    #     }
    # },
    # This is recurring concept drift rather than an early learning curve on a stationary stream, so not very useful for this experiment

    # {
    # 'stream_type': 'Waveform',
    #  'stream' : {
    #     'has_noise': True
    #     }
    # },
    # This is recurring concept drift rather than an early learning curve on a stationary stream, so not very useful for this experiment

    # {
    # 'stream_type': 'Friedman',
    #  'stream' : {
    #     }
    # },
    # This stream is regression not classification

    # {
    # 'stream_type': 'Mv',
    #  'stream' : {
    #     }
    # },
    # This stream is regression not classification (plus freezes in a loop)

    # {
    # 'stream_type': 'STAGGER',
    #  'stream' : {
    #     'classification_function': 0,
    #     'balance_classes': False,
    #     }
    # },
    ## Most results (9/10) are 1.0, too trivial a data stream to learn 


]

