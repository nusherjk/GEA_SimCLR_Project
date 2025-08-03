from load_cifer10 import load_cifar10
data_loc='../cifardata/'
api_loc ='~/nas_benchmark_datasets/NAS-Bench-201-v1_1-096897.pth'
save_loc= 'results/'
# parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
batch_size = 256
evaluate_size= 256
repeat=1
augtype = None
sigma=0.05
# parser.add_argument('--GPU', default='0', type=str)
# parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
# parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
# parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
# parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
# parser.add_argument('--seed', default=1, type=int)
# parser.add_argument('--trainval', action='store_true')
# parser.add_argument('--dataset', default='cifar10', type=str)
# parser.add_argument('--n_samples', default=100, type=int)
# parser.add_argument('--n_runs', default=1, type=int)
# parser.add_argument('--regularize', default='oldest', type=str, help='which scheme to use to remove indvs from population',
                    # choices=["oldest", "highest", "lowest"])
# parser.add_argument('--sampling', default='S', type=str, help='which scheme to use to sample candidates to be parent',
#                     choices=["S", "highest", "lowest"])
# parser.add_argument('--C', default=200, type=int)
# parser.add_argument('--P', default=10, type=int)
# parser.add_argument('--S', default=5, type=int)




C = 200
P = 10
S = 5

sampling = S



population = [] # Polpulation < -- empty loop
history = [] # history <-- empty
train_loader, test_loader = load_cifar10()

while len(population) < C:
    # create random architecture


    # calculate the proxy Score

    # push model to the right
    population.append()
