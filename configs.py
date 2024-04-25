
BASE_CONFIG = {
    'change_point_epoch': 10,
    'evaluation_interval': 1000,
    'num_epochs': 20,
    'delta_easy': 1e-3,
    'delta_hard': 1e-7,
    'seed0': 0,
    'seed1': 100,
    'update_delta_dropped_accuracy': 1.0,
    'num_runs': 2,
    'model': 'UpdatableHoeffdingTreeClassifier',
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
    'actions': {
        # Actions to change delta_hard are a multiplier list from 1/100 to 100, with 1 meaning no change
        'delta_move':  [1/100, 1/10, 1, 10, 100],
    },
    'num_episodes': 10
}

STREAMS = [

    {
    'stream_type': 'RandomTree',
     'stream' : {
        'n_classes': 2,
        'n_num_features': 2,
        'n_cat_features': 2,
        'n_categories_per_feature': 2,
        'max_tree_depth': 5,
        'first_leaf_level': 2,
        'fraction_leaves_per_level': 0.15,
        }
    },


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

    {
    'stream_type': 'RandomTree',
     'stream' : {
        'n_classes': 4,
        'n_num_features': 4,
        'n_cat_features': 4,
        'n_categories_per_feature': 4,
        'max_tree_depth': 5,
        'first_leaf_level': 2,
        'fraction_leaves_per_level': 0.15,
        }
    },

    {
    'stream_type': 'RandomTree',
     'stream' : {
        'n_classes': 5,
        'n_num_features': 5,
        'n_cat_features': 5,
        'n_categories_per_feature': 5,
        'max_tree_depth': 5,
        'first_leaf_level': 2,
        'fraction_leaves_per_level': 0.15,
        }
    },


    {
    'stream_type': 'RandomRBF',
     'stream' : {
        'n_classes': 4,
        'n_features': 5,
        'n_centroids': 10,
        }
    }

    {
    'stream_type': 'RandomRBF',
     'stream' : {
        'n_classes': 2,
        'n_features': 10,
        'n_centroids': 50,
        }
    },


    {
    'stream_type': 'RandomRBF',
     'stream' : {
        'n_classes': 4,
        'n_features': 20,
        'n_centroids': 100,
        }
    }

    {
    'stream_type': 'Hyperplane',
     'stream' : {
         'n_features': 10,
         'noise_percentage': 0.05,
         'sigma': 0.1,
        }
    }

]

