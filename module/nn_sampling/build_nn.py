from module.nn_sampling.utils.torch_probabilities import CustomLoss, custom_loss_distribution
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################Build Model##################################

class NeuralNetwork(nn.Module):
    def __init__(self, config={}, target_distribution=[], width=60, depth=4, lr=0.001, dist="kl", n_samples=5000, opt=1, scale_blocks_with_n_sources=False):
        super(NeuralNetwork, self).__init__()
        self.config = config
        self.target_distribution = target_distribution
        self.width=width
        self.depth=depth
        self.lr=lr
        self.opt=opt
        self.dist = dist #type of distance, should be "kl" or "eucl". Define other distance in torch_probabilities.py
        self.scale_blocks_with_n_sources = scale_blocks_with_n_sources #option to have the block of a party with width multiplied by the number of sources connected

        self.input_width = nn.ModuleDict()
        self.hidden_width = nn.ModuleDict()
        self.output_width = nn.ModuleDict()
        self.best_dist = float('inf')
        self.n_samples = n_samples if n_samples is not None else 5000

        self.sources = set(src for party in config for src in config[party][0])
        self.source_indices = {source: idx for idx, source in enumerate(self.sources)}
        try:
            self.build_model()
        except:
            pass

    def build_model(self):
        #reinitialize
        self.input_width = nn.ModuleDict()
        self.hidden_width = nn.ModuleDict()
        self.output_width = nn.ModuleDict()
        
        self.sources = set(src for party in self.config for src in self.config[party][0])
        self.source_indices = {source: idx for idx, source in enumerate(self.sources)}

        for party, (sources, value) in self.config.items():
            input_dim = len(sources)
            
            if self.scale_blocks_with_n_sources:
                hidden_layer_size = self.width * len(self.sources)
            else:
                hidden_layer_size = self.width
                
            
            self.input_width[party] = nn.Linear(input_dim, hidden_layer_size)
            
            hidden_width = []
            for _ in range(self.depth):
                hidden_width.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                hidden_width.append(nn.ReLU())
            hidden_width.append(nn.Linear(hidden_layer_size, value))
            
            self.hidden_width[party] = nn.Sequential(*hidden_width)
            self.output_width[party] = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        outputs = []
        for party, layer in self.input_width.items():
            input_indices = [self.source_indices[source] for source in self.config[party][0]]
            selected_inputs = x[:, input_indices]
            h = layer(selected_inputs)
            h = self.hidden_width[party](h)
            out = self.output_width[party](h)
            outputs.append(out)

        joint_prob = torch.cat(outputs, dim=1)
        return joint_prob
    
    def outputsize(self):
        keys= list(self.config.keys())
        n_outputs = 0
        for akey in keys:
            n_outputs+= self.config[akey][1]
        return n_outputs


#############################Generate Data#####################################



def generate_data(config, n_samples, target_distribution=[], opt=1):
    """generate the input for the neural network, different distribution available"""
    data_type = np.float32 
    
    sources = list(set(source for person in config for source in config[person][0]))
    n_sources = len(sources)
    
    if opt == 1:
        data = np.random.randn(n_samples, n_sources)
        data = np.divide(data, 0.1443375673).astype(data_type)
        
    elif opt == 2 :
        data = np.random.uniform(0, 1, (n_samples, n_sources)).astype(data_type)
    else:
        raise ValueError("Invalid option for 'opt'. Use 1 for normalized standard normal distribution, 2 for adjusted distribution with mean -0.5 and variance 0.28867513459, 3 for adjusted distribution with mean -0.5 and variance 0.1443375673, or 4 for uniform distribution between -20 and 20.")
    
    labels = np.tile(target_distribution, (n_samples, 1)).astype(data_type)
    
    return data, labels


################################Train Model########################


def sampling(loss, n_output, dist, bias=4, min_sampling=1000, max_sampling=float("inf")):
    """return the optimal number for sampling for a given loss and number of outputs"""
    ## found by plotting kl divergence of sampled distribution for 64 outputs
    ## the sampling is linear with the number of outputs
    ## the bias tells how precise the distribution should be 
    def func_kl(x):
        A=1
        B=2
        return 1/(B*x**A)
    def func_eucl(x):
        A=2
        B=1
        return 1/(B*x**A)
    
    if dist=="kl":
        out = min(max(int(func_kl(loss/bias)*n_output), min_sampling), max_sampling)
    elif dist=="eucl":
        out = min(max(int(func_eucl(loss/bias)), min_sampling), max_sampling)
    return out

def train_model_and_save(model, n_samples_in=None, n_training_steps=10000, model_path='model/model.pth', threshold=0, bias=4, n_steps_reevaluate=10000, bias_max=10, min_sampling = 1000, max_sampling=float("inf")):
    """main function for training
    
    Parameters:
        - model : an instance of NeuralNetwork
        - n_samples_in=None : the number of samples for training. If None, the number of samples will adapt as a function of the loss
        - n_training_steps=10000 : the number of training steps maximal
        - model_path='model/model.pth' : the path to save the model
        - threshold=0 : distance when stopping the training
        - bias=4 : if n_samples_in=None, control the number of samples given the loss
        - n_steps_reevaluate=10000 : reevaluate the model if no improvement is found after a number of training steps, increase the bias when reevaluating
        - bias_max=10 : fix the maximal bias increased when reevaluating when n_steps_reevaluate passed without update
        - min_sampling=1000 : if n_samples_in=None, minimum number of samples
        - max_sampling=float("inf") : if n_samples_in=None, fix the maximum number of samples
    
    Returns:
        model.best_dist : the best distance found during training
    """
    
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model.to(device)
    
    if n_samples_in is not None:
        model.n_samples=n_samples_in
    
    criterion = CustomLoss(model.config, dist=model.dist)
    np.set_printoptions(precision=8, suppress=False)
    
    #fonction to generate the distribution from torch_probabilities, compiled for speed (no improvement)
    compiled_distribution_function = torch.compile(custom_loss_distribution)
        
    n_outputs = len(model.target_distribution)
    
    n_epochs_no_update = 0
    
    with tqdm(range(n_training_steps)) as t:
        for epoch in t:
            
            model.train()
    
            data, labels = generate_data(model.config, model.n_samples, model.target_distribution, model.opt)
            # data, labels = torch.tensor(data).clone().detach().to(device), torch.tensor(labels).clone().detach().to(device)
            data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)
    
            total_outputs = model(data)
            model.optimizer.zero_grad()
            loss, predict= criterion(total_outputs, labels, compiled_distribution_function)
            
            if epoch != 0:
                loss.backward()
                model.optimizer.step()
            
        
            model.eval()
            with torch.no_grad():
                ## automatic change of the sampling (model.n_samples) with the loss
                if loss < model.best_dist:
                    data, labels = generate_data(model.config, model.n_samples, model.target_distribution, model.opt)
                    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)
                    total_outputs = model(data)
                    reeval_loss, _ = criterion(total_outputs, labels, compiled_distribution_function)
                    if reeval_loss < 1.1*loss:
                        model.best_dist = (loss + reeval_loss)/2
                        
                        elapsed = t.format_dict['elapsed']
                        elapsed_str = t.format_interval(elapsed)
            
                        print(f"update model : dist ({model.dist}) =", np.around(float(loss), 5), "time ", elapsed_str, "n_samples", model.n_samples, "epoch", epoch)
                        save_model_and_optimizer(model, model_path)
                        n_epochs_no_update = 0
                        
                        #update the number of samples
                        if n_samples_in is None:
                            model.n_samples= sampling(loss, n_outputs, model.dist, bias, min_sampling, max_sampling)
                    
                else:
                    ## reevaluation of the best model
                    n_epochs_no_update += 1
                    if n_epochs_no_update > n_steps_reevaluate:
                        model.best_dist = evaluate_model_return_dist(model.n_samples, model, dist=model.dist)
                        n_epochs_no_update = 0
                        
                        if bias<bias_max:
                            bias+=1 #increase the sampling to end the training and making sure we are not under sampling
                        print("reevaluation of the model : {}".format(model.best_dist))

                if model.best_dist<threshold:
                    # model.best_dist = evaluate_model_return_dist(model, model_path, model.n_samples, model.target_distribution, model.opt, model.dist)
                    # print("reevaluation of the model because {}<{}: new evaluation = {}".format(loss, threshold, model.best_dist))
                    # if model.best_dist<threshold:
                    print("stopping the training: reached the threshold of {}".format(threshold))
                    break

                # torch.cuda.empty_cache()

    try:
        print(f"Loss ({model.dist}): {model.best_dist}\n ")
    except:
        print("No best result found")
        
    return model.best_dist



####################################Save and Load################################

def save_model_and_optimizer(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'best_dist': model.best_dist,
        'target_distribution': model.target_distribution,
        'width': model.width,
        'depth': model.depth,
        'opt': model.opt,
        'config': model.config,
        'lr': model.lr,
        'dist': model.dist,
        'scale_blocks_with_n_sources': model.scale_blocks_with_n_sources
    }, path)

def load_model_and_optimizer(model, path='model/model.pth', print_in_console=True):
    model.to(device)
    checkpoint = torch.load(path, map_location=device)
    
    model.best_dist = checkpoint.get('best_dist', float('inf'))
    model.target_distribution = checkpoint.get('target_distribution')
    model.width = checkpoint.get('width', int(60))
    model.depth = checkpoint.get('depth', int(4))
    model.opt = checkpoint.get('opt', 1)
    model.config = checkpoint.get('config')
    model.lr = checkpoint.get('lr')
    model.dist = checkpoint.get('dist')
    model.scale_blocks_with_n_sources=checkpoint.get('scale_blocks_with_n_sources')

    model.build_model()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if print_in_console:
        print("Model and optimizers loaded from :", path)
        print("Best distance {} loaded :".format(model.dist), model.best_dist)

def load_model_without_checkpoints(model, n_samples, path='model/model.pth', dist =None, print_in_console=True):
    model.to(device)
    checkpoint = torch.load(path, map_location=device)

    model.target_distribution = checkpoint.get('target_distribution')
    model.width = checkpoint.get('width', int(60))
    model.depth = checkpoint.get('depth', int(4))
    model.opt = checkpoint.get('opt', 1)
    model.config = checkpoint.get('config')
    model.lr = checkpoint.get('lr')
    model.dist = checkpoint.get('dist')
    model.scale_blocks_with_n_sources=checkpoint.get('scale_blocks_with_n_sources')

    model.build_model()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if dist is not None:
        model.dist = dist

    model.best_dist = evaluate_model_return_dist(n_samples, model, path, dist=model.dist)
    
    if print_in_console:
        print("Model and optimizers loaded from :", path)
        print("Best distance {} evaluated :".format(model.dist), model.best_dist)

    
####################################Evaluate######################################


def get_dist_from_model(path='model/model.pth'):
    """return the distance of a saved model from a path
    inputs : path of saved model
    outputs : distance"""
    model=NeuralNetwork()
    load_model_and_optimizer(model, path, print_in_console=False)

    return model.best_dist


def evaluate_model(n_samples, model=None, path=None):
    """evaluate a model and return the distance and probability distribution
    One preloaded model or path to a model to load should be given
    inputs: 
        - n_samples: the number of samples
        - model=None: a preloaded model
        - path=None: a path to a model to load
    outputs:
        - the distance
        - the probability distribution
        """
    
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: evaluate_model(n_samples, model=None, path=None) "))

    criterion = CustomLoss(model.config, dist=model.dist)
    model.eval() 
    with torch.no_grad():
        data, labels = generate_data(model.config, n_samples, model.target_distribution, model.opt)
        data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

        total_outputs = model(data)

        d, predict = criterion(total_outputs, labels)
    
    return d, predict

def evaluate_model_return_distribution(n_samples, model=None, path=None):
    """evaluate a model and return the probability distribution
    One preloaded model or path to a model to load should be given
    inputs: 
        - n_samples: the number of samples
        - model=None: a preloaded model
        - path=None: a path to a model to load
    outputs:
        - the probability distribution
        """    
        
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: evaluate_model_return_distribution(n_samples, model=None, path=None) "))

    criterion = CustomLoss(model.config, dist=model.dist)
    model.eval() 
    
    with torch.no_grad():
        data, labels = generate_data(model.config, n_samples, model.target_distribution, model.opt)
        data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

        total_outputs = model(data)

        d, predict = criterion(total_outputs, labels)
    
    return predict.detach().cpu().numpy()

    
def evaluate_model_return_dist(n_samples, model=None, path=None, dist=None):
    """evaluate a model and return the distance
    One preloaded model or path to a model to load should be given
    inputs: 
        - n_samples: the number of samples
        - model=None: a preloaded model
        - path=None: a path to a model to load
        - dist=None: the distance used (if None is given, the distance model.dist will be used)
    outputs:
        - the distance
        """
        
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: evaluate_model_return_dist(n_samples, model=None, path=None, dist=None) "))

    if dist is not None:
        model.dist = dist
    
    criterion = CustomLoss(model.config, model.dist)
    model.eval() 

    with torch.no_grad():
        data, labels = generate_data(model.config, n_samples, model.target_distribution, model.opt)
        data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

        total_outputs = model(data)
        d, predict = criterion(total_outputs, labels)
    return d


def output_model(model, n_samples):
    """gives the output of the neural network, without computing the final distribution"""
    model.eval()
    
    data, _ = generate_data(model.config, n_samples, model.target_distribution, model.opt)
    data = torch.tensor(data).to(device)

    with torch.no_grad(): 
        outputs = model(data)

    separated_outputs = []
    start_idx = 0
    for party, (_, output_size) in model.config.items():
        end_idx = start_idx + output_size
        separated_outputs.append(outputs[:, start_idx:end_idx])
        start_idx = end_idx

    grouped_outputs = torch.stack(separated_outputs, dim=1)

    return data, grouped_outputs

def plot_distribution(n_samples, model=None, path=None, dist=None, title_style=1):
    """function to plot a scatter graph with the target distribution (orange), and local distribution (blue)
    inputs:
        - n_samples
        - model=None : preloaded model. If not given, a path to a model should be given
        - path=None : path to a model. If not given, a preloaded model should be given
        - dist=None : distance to evaluate the difference between the sampled and target distributions. If not given, the model.dist will be taken
        - title_style=1 : 0 for no title, 1 for path (if given) and distance, 2 for distance only
    outputs:
        no outputs, print the graph in console"""
        
    import matplotlib.pyplot as plt
    
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: plot_distribution(n_samples, model=None, path=None, dist=None, title_style=1)"))

    if dist is not None:
        model.dist=dist
    criterion = CustomLoss(model.config, model.dist)
    model.eval() 

    data, labels = generate_data(model.config, n_samples, model.target_distribution, model.opt)
    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

    with torch.no_grad():
        total_outputs = model(data)
        d, predict = criterion(total_outputs, labels)
    
    
    plt.scatter(np.arange(len(predict)), predict.detach().numpy())
    plt.scatter(np.arange(len(model.target_distribution)), model.target_distribution)
    
    if title_style==0: 
        pass
    elif title_style==1:
        if path is None:
            path=""
        plt.title(path + "  distance ({}) = {}".format(model.dist, '{0:.6f}'.format(d)))
    elif title_style==2:
        plt.title("distance ({}) = {}".format(model.dist, '{0:.6f}'.format(d)))
    else:
        print("title_style not defined")
    plt.plot()


####################################Plot strategies######################################

def output_model_with_some_fixed_inputs(model, n_samples, sources, sources_to_vary, set_sources_index):
    """same as output_model but we fix the value of some input to only vary two inputs (for plotting purposes)"""
    model.eval()
    
    data, _ = generate_data(model.config, n_samples, model.opt)
    data = torch.tensor(data).to(device)
    
    ##fixing sources to fix
    for item in set_sources_index:
        data[:,item]=set_sources_index[item]
            
    with torch.no_grad(): 
        outputs = model(data)

    separated_outputs = []
    start_idx = 0
    for party, (_, output_size) in model.config.items():
        end_idx = start_idx + output_size
        separated_outputs.append(outputs[:, start_idx:end_idx])
        start_idx = end_idx

    return data, separated_outputs


def create_tuples_excluding_index(values_list_input, values_list_output, index, source1, source2):
    if not isinstance(values_list_input, torch.Tensor):
        values_list_input = torch.tensor(values_list_input).clone().detach()
    else:
        values_list_input = values_list_input.clone().detach()
    
    if not isinstance(values_list_output, torch.Tensor):
        values_list_output = torch.tensor(values_list_output).clone().detach()
    else:
        values_list_output = values_list_output.clone().detach()

    source = torch.zeros(values_list_input.shape[1], dtype=torch.bool)
    source[source1] = True
    source[source2] = True
    
    result_tuples = values_list_input[:, source]
    
    if source1>source2: #inverse axes x and y
        result_tuples=torch.concat((result_tuples[:,1:], result_tuples[:,:1]), dim=1)
        
    selected_output = values_list_output[index]

    return result_tuples, selected_output

def output_model_separated(model, n_samples):

    model.eval()
    
    data, _ = generate_data(model.config, n_samples, opt=model.opt)
    data = torch.tensor(data).to('cpu')

    with torch.no_grad(): 
        outputs = model(data)

    separated_outputs = []
    start_idx = 0
    for party, (_, output_size) in model.config.items():
        end_idx = start_idx + output_size
        separated_outputs.append(outputs[:, start_idx:end_idx])
        start_idx = end_idx

    # grouped_outputs = torch.stack(separated_outputs, dim=1)
    
    return data, separated_outputs

def plot(coordinates, values_list, xlabel, ylabel, title, plot_most_likely_outcomes=False):
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(3,3), dpi=400)
    # colors = ['#47a9ff', '#ff8d47', '#5cff47', '#ff474d']
    colors_default = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    colors_vibrant_fusion = ["#147df5", "#ff8700", "#deff0a", "#0aff99", "#ff0000", "#ffd300", "#a1ff0a", "#0aefff"]
    n_colors = len(values_list[0])
    colors=[]
    colormap=plt.cm.get_cmap("gist_ncar",n_colors+1)
    
    for i in range(n_colors):
        # colors.append(colors_default[i])
        colors.append(colors_vibrant_fusion[i])
        # colors.append(colormap(i))
    
    if plot_most_likely_outcomes:
        max_indices = torch.argmax(values_list, dim=1) 
    else:
        max_indices = torch.reshape(torch.multinomial(values_list, 1), [torch.multinomial(values_list, 1).shape[0],])

    assigned_colors = [colors[idx] for idx in max_indices]
    
    for (x, y), color in zip(coordinates, assigned_colors):
        plt.scatter(x, y, alpha=1, linewidth=0, color=color, label= color if color not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.show()


def gaussian_to_uniform(x,mu, sigma):
    """map a gaussian distribution to the interval [0,1]"""
    from scipy import special
    return 1/2*(1+special.erf((x-mu)/(sigma*np.sqrt(2))))

def zero_one_interval_to_gaussian(y, mu, sigma):
    """inverse of the function "gaussian_to_uniform"
    map the interval [0,1] to a gaussian distribution with mean mu and variance sigma"""
    from scipy import special
    return np.sqrt(2)*sigma*special.erfinv(2*y-1)+mu
    
    
def plot_strats(model, party, source_x, source_y, n_points=1000, set_sources_value={}, uniform_local_variables=True, plot_most_likely_outcomes=False):
    """output the strategy of a party when changing the local variable of two sources
    (this only works for network with parties connected to two sources)
     
    Inputs:
         - model : instance of NeuralNetwork with right weight loaded
         - party : name of the party, the same as in model.config
         - source_x : the name of the source plotted in the x axis, as in model.config
         - source_y : the name of the source plotted in the y axis, as in model.config
         - n_points=1000 : the number of samples plotted
         
    Outputs:
        - The values of the local variables
        - The outcome of each party for each values of the local variables  
    """
    ## set find source index from model.config
    sources = list(set(source for person in model.config for source in model.config[person][0]))
    source1 = sources.index(source_x)
    source2 = sources.index(source_y)
        
    parties=[]
    for aparty in model.config:
        parties.append(aparty)
    index = parties.index(party)
    
    ## get local variables and parties outputs
    opt=model.opt
    if opt==1:
        mu, sigma = 0, 1/0.1443375673

        set_sources_index={}
        for item in set_sources_value:
            set_sources_index[sources.index(item)]=zero_one_interval_to_gaussian(set_sources_value[item], mu, sigma)
            
        data, values_list_output = output_model_with_some_fixed_inputs(model, n_points, sources, [source1, source2], set_sources_index)
        if uniform_local_variables:
            data = gaussian_to_uniform(data, mu, sigma)

    else:
        raise(ValueError) #define other opt for plotting other distributions than opt=1

    values_list_output2 = torch.stack(values_list_output, dim=0)

    coordinates, values_list = create_tuples_excluding_index(data, values_list_output2, index=index,source1=source1,source2=source2)

    plot(coordinates, values_list, xlabel=source_x, ylabel=source_y, title="Local strategy of "+party, plot_most_likely_outcomes=plot_most_likely_outcomes)
    return data, values_list_output2


    
    
    





