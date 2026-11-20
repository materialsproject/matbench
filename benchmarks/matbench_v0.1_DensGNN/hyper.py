hyper_1 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {

                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        # "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128,        
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': None,
                                           'max_voronoi_area': None}},


                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", "g_pooling_method":"max",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },
    "training": {
            "fit": {"batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},

            "compile": {

                # "optimizer": {"class_name": "Adam",
                #     "config": {"lr": {
                #                     "class_name": "ExponentialDecay",
                #                     "config": {"initial_learning_rate": 0.001,
                #                     "decay_steps": 5800,
                #                     "decay_rate": 0.5, "staircase":  False},
                #         }
                #     }
                # },

                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                            "config": {
                                "dataset_size": 509, "batch_size": 64, "epochs": 800,
                                "lr_start": 0.0001, "lr_stop": 1.0e-05
                            }
                        }
                    }
                },


                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        
    "data": {
            "dataset": {
                "config": {},
                "methods": [
               
                    {"set_representation": {
                        "pre_processor": {

                            "class_name": "KNNUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 24}

                            # "class_name": "VoronoiUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"min_ridge_area": 0.1}

                                          },
                        "reset_graphs": False}},

                ]
            },
            "data_unit": ""
        },
       
    
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}

hyper_2 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {
                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        # "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128,        
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': None,
                                           'max_voronoi_area': None}},


                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"max", "g_pooling_method":"max",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },

    "training": {
            "fit": {"batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                                    "class_name": "ExponentialDecay",
                                    "config": {"initial_learning_rate": 0.001,
                                    "decay_steps": 5800,
                                    "decay_rate": 0.5, "staircase":  False},
                        }
                    }
                },

                #  "optimizer": {
                #     "class_name": "Adam",
                #     "config": {
                #         "learning_rate": {
                #             "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                #             "config": {
                #                 "dataset_size": 1013, "batch_size": 64, "epochs": 800,
                #                 "lr_start": 0.0001, "lr_stop": 1.0e-05
                #             }
                #         }
                #     }
                # },
                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        
    "data": {
            "dataset": {
                "config": {},
                "methods": [
               
                    {"set_representation": {
                        "pre_processor": {

                            "class_name": "KNNUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 12}

                            # "class_name": "VoronoiUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"min_ridge_area": 0.1}

                                          },
                        "reset_graphs": False}},

                ]
            },
            "data_unit": ""
        },
       
    
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}


hyper_3 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {
                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128,  
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': 32,
                                           'max_voronoi_area': 32}},

                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", "g_pooling_method":"max",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },
    "training": {
            "fit": {"batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},

            "compile": {

                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                                    "class_name": "ExponentialDecay",
                                    "config": {"initial_learning_rate": 0.001,
                                    "decay_steps": 5800,
                                    "decay_rate": 0.5, "staircase":  False},
                        }
                    }
                },
                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
    "data": {
            "dataset": {
                "config": {},
                "methods": [
               
                    {"set_representation": {
                        "pre_processor": {
                            # "class_name": "KNNUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"k": 12}
                            "class_name": "VoronoiUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"min_ridge_area": 0.1}
                                          },
                        "reset_graphs": False}},
                ]
            },
            "data_unit": ""
        },
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}

hyper_4 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {
                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           },
                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128, 
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': 32,
                                           'max_voronoi_area': 32}},

                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", "g_pooling_method":"max",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },

    "training": {
            "fit": {"batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},

            "compile": {

                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                                    "class_name": "ExponentialDecay",
                                    "config": {"initial_learning_rate": 0.001,
                                    "decay_steps": 5800,
                                    "decay_rate": 0.5, "staircase":  False},
                        }
                    }
                },
                # "optimizer": {
                #     "class_name": "Adam",
                #         "config": {
                #             "learning_rate": {
                #                 "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                #                 "config": {
                #                     "dataset_size": 15142, "batch_size": 64, "epochs": 800,
                #                     "lr_start": 0.0005, "lr_stop": 1.0e-05
                #                 }
                #             }
                #         }
                # },
                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        
    "data": {
            "dataset": {

                "config": {},
                "methods": [
               
                    {"set_representation": {
                        "pre_processor": {

                            # "class_name": "KNNUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"k": 12}

                            "class_name": "VoronoiUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"min_ridge_area": 0.1}

                                          },
                        "reset_graphs": False}},

                ]
            },
            "data_unit": ""
        },
       
    
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}


hyper_5 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {

                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
                           },
                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128,      
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': 32,
                                           'max_voronoi_area': 32}},


                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", "g_pooling_method":"max",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },
    "training": {
            "fit": {"batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                                    "class_name": "ExponentialDecay",
                                    "config": {"initial_learning_rate": 0.001,
                                    "decay_steps": 5800,
                                    "decay_rate": 0.5, "staircase":  False},
                        }
                    }
                },
                #  "optimizer": {
                #     "class_name": "Adam",
                #     "config": {
                #         "learning_rate": {
                #             "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                #             "config": {
                #                 "dataset_size": 8789, "batch_size": 64, "epochs": 800,
                #                 "lr_start": 0.0001, "lr_stop": 1.0e-05
                #             }
                #         }
                #     }
                # },

                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        
    "data": {
            "dataset": {
                "config": {},
                "methods": [
                    {"set_representation": {
                        "pre_processor": {

                            # "class_name": "KNNUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"k": 12}

                            "class_name": "VoronoiUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"min_ridge_area": 0.1}

                                          },
                        "reset_graphs": False}},

                ]
            },
            "data_unit": ""
        },
       
    
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}


hyper_6 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {
                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128,       
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': 32,
                                           'max_voronoi_area': 32}},

                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", "g_pooling_method":"mean",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },

    "training": {
            "fit": {"batch_size": 128, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},

            "compile": {

                "optimizer": {"class_name": "Adam",
                    "config": {"lr": {
                                    "class_name": "ExponentialDecay",
                                    "config": {"initial_learning_rate": 0.001,
                                    "decay_steps": 5800,
                                    "decay_rate": 0.5, "staircase":  False},
                        }
                    }
                },

                #  "optimizer": {
                #     "class_name": "Adam",
                #     "config": {
                #         "learning_rate": {
                #             "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                #             "config": {
                #                 "dataset_size": 8789, "batch_size": 64, "epochs": 800,
                #                 "lr_start": 0.0001, "lr_stop": 1.0e-05
                #             }
                #         }
                #     }
                # },

                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        
    "data": {
            "dataset": {
                "config": {},
                "methods": [
                    {"set_representation": {
                        "pre_processor": {

                            # "class_name": "KNNUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"k": 12}

                            "class_name": "VoronoiUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"min_ridge_area": 0.1}

                                          },
                        "reset_graphs": False}},

                ]
            },
            "data_unit": ""
        },
       
    
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}


hyper_7 = {
    "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {

                        "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                        # "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                        "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                        # "AGNIFinger": {"shape": (None,61), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "AGNIFinger": {"shape": (None,24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                        "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                        "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                           
                           },

                "input_block_cfg" : {'node_size': 128,
                   'edge_size': 128,       
                   'edge_embedding_args': {'bins_distance': 32,
                                           'max_distance': 8.0,
                                           'distance_log_base': 1.0,
                                           'bins_voronoi_area': None,
                                           'max_voronoi_area': None}},


                "output_block_cfg" : {'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {'units': [1],
                                   'activation': ['linear']},
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {'units': [32, 1],
                                                 'activation': ['swish', 'swish']},
                    'edge_attention_mlp_global': {'units': [32, 1],
                                                  'activation': ['swish', 'swish']},
                    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],  
                    'multiplicity_readout': True},
                    
                "input_embedding": {"node": {"input_dim": 96, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}
                                    },
                "depth": 5,
                "n_units":128, 
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },
                "graph_mlp": {"units": [128], "use_bias": True, "activation": ["swish"],
                            },

                "gin_args": {"pooling_method":"sum", "g_pooling_method":"max",
                             "edge_mlp_args": {"units": [128]*3, "use_bias": True, "activation": ["swish"]*3}, 
                             "concat_args": {"axis": -1}, 
                             "node_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             "graph_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                             },
            }
        },

    "training": {
            "fit": {"batch_size": 64, "epochs": 300, "validation_freq": 20, "verbose": 2, "callbacks": []},
            "compile": {

              # "optimizer": {"class_name": "Adam",
                #     "config": {"lr": {
                #                     "class_name": "ExponentialDecay",
                #                     "config": {"initial_learning_rate": 0.001,
                #                     "decay_steps": 5800,
                #                     "decay_rate": 0.5, "staircase":  False},
                #         }
                #     }
                # },

                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>KerasPolynomialDecaySchedule",
                            "config": {
                                "dataset_size": 3811, "batch_size": 64, "epochs": 800,
                                "lr_start": 0.0001, "lr_stop": 1.0e-05
                            }
                        }
                    }
                },

                "loss": "mean_absolute_error"
            },

            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
            "scaler": {"class_name": "StandardScaler", "config": {"with_std": True, "with_mean": True, "copy": True}}
        },
        
    "data": {
            "dataset": {
                "config": {},
                "methods": [
                    {"set_representation": {
                        "pre_processor": {
                            "class_name": "KNNUnitCell",
                                          "module_name": "kgcnn.crystal.preprocessor",
                                          "config": {"k": 12}

                            # "class_name": "VoronoiUnitCell",
                            #               "module_name": "kgcnn.crystal.preprocessor",
                            #               "config": {"min_ridge_area": 0.1}

                                          },
                        "reset_graphs": False}},
                ]
            },
            "data_unit": ""
        },
       
    
    "info": {
        "postfix": "",
        "postfix_file": "",
        "kgcnn_version": "3.0.2"
    }
}
