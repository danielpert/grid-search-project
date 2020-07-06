import signac
import itertools

# initialize project
project = signac.init_project('grid-search-project')

# define grid of values to search
max_features = ['log2', 'sqrt']
max_depth = [None, 20, 40, 60, 80]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 5]

var_threshold = [0.01, 0.02, 0.03]
corr_threshold = [0.8, 0.9, 0.95]

for combin in itertools.product(max_features, max_depth, min_samples_split, \
        min_samples_leaf, var_threshold, corr_threshold):
    sp = {'max_features': combin[0],
          'max_depth': combin[1],
          'min_samples_split': combin[2],
          'min_samples_leaf': combin[3],
          'var_threshold': combin[4],
          'corr_threshold': combin[5]}
    job = project.open_job(sp)
    job.init()
