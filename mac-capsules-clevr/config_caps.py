import os
import argparse

###################################### configuration ######################################
class Config(object):

    typeFilters = [[], ["1_query_size_",
                "1_query_material_",
                "2_equal_color_",
                "2_equal_shape_"],
                ["1_query_color_",
                "1_query_shape_",
                "2_equal_size_",
                "2_equal_material_"]]

    #### files interface
    ## data files
    dataPath = "" # dataset specific
    datasetFilename = "" # dataset specific

    # file names
    imagesFilename = "{tier}.h5" # Images
    instancesFilename = "{tier}Instances.json"
    # symbols dictionaries
    questionDictFilename = "questionDict.pkl"
    answerDictFilename = "answerDict.pkl"
    qaDictFilename = "qaDict.pkl"
    
    ## experiment files
    expPathname = "{expName}"
    expName = "" #  will be assigned through argparse

    weightsPath = "./weights"
    weightsFilename = "weights{epoch}.ckpt"

    # model predictions and optionally attention maps
    predsPath = "./preds"
    predsFilename = "{tier}Predictions-{expName}.json"
    answersFilename = "{tier}Answers-{expName}.txt"

    # logging of accuracy, loss etc. per epoch
    logPath = "./results"
    logFilename = "results-{expName}.csv"

    # configuration file of the used flags to run the experiment
    configPath = "./results"
    configFilename = "config-{expName}.json"

    def toString(self):
        return self.expName

    # make directories of experiment if not exist yet
    def makedirs(self, directory):
        directory = os.path.join(directory, self.expPath())
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    ### filename builders
    ## data files
    def dataFile(self, filename): 
        return os.path.join(self.dataPath, filename)

    def generatedFile(self, filename): 
        return self.dataFile(self.generatedPrefix + filename)

    datasetFile     = lambda self, tier: self.dataFile(self.datasetFilename.format(tier = tier))
    imagesIdsFile   = lambda self, tier: self.dataFile(self.imgIdsFilename.format(tier = tier)) #
    imagesFile      = lambda self, tier: self.dataFile(self.imagesFilename.format(tier = tier))
    instancesFile   = lambda self, tier: self.generatedFile(self.instancesFilename.format(tier = tier))

    questionDictFile    = lambda self: self.generatedFile(self.questionDictFilename)
    answerDictFile      = lambda self: self.generatedFile(self.answerDictFilename)
    qaDictFile          = lambda self: self.generatedFile(self.qaDictFilename)

    ## experiment files
    expPath     = lambda self: self.expPathname.format(expName = self.toString())
    
    weightsDir  = lambda self: self.makedirs(self.weightsPath)
    predsDir    = lambda self: self.makedirs(self.predsPath)
    logDir      = lambda self: self.makedirs(self.logPath)
    configDir   = lambda self: self.makedirs(self.configPath)

    weightsFile     = lambda self, epoch: os.path.join(self.weightsDir(), self.weightsFilename.format(epoch = str(epoch)))
    predsFile       = lambda self, tier: os.path.join(self.predsDir(), self.predsFilename.format(tier = tier, expName = self.expName))
    answersFile     = lambda self, tier: os.path.join(self.predsDir(), self.answersFilename.format(tier = tier, expName = self.expName))
    logFile         = lambda self: os.path.join(self.logDir(), self.logFilename.format(expName = self.expName))
    configFile      = lambda self: os.path.join(self.configDir(), self.configFilename.format(expName = self.expName))


# global configuration variable. Holds file paths and program parameters
config = Config()

###################################### arguments ######################################
def parseArgs():
    parser = argparse.ArgumentParser(fromfile_prefix_chars = "@")

    ################ systems

    # gpus and memory
    parser.add_argument("--gpus",           default = "", type = str,       help = "comma-separated list of gpus to use")
    parser.add_argument("--gpusNum",        default = 1, type = int,        help = "number of gpus to use")
    
    parser.add_argument("--allowGrowth",    action = "store_true",          help = "allow gpu memory growth")
    parser.add_argument("--maxMemory",      default = 1.0, type = float,    help = "set maximum gpu memory usage")
    
    parser.add_argument("--parallel",       action = "store_true",          help = "load images in parallel to batch running")
    parser.add_argument("--workers",        default = 1, type = int,        help = "number of workers to load images")
    parser.add_argument("--taskSize",       default = 8, type = int,        help = "number of image batches to load in advance") # 40
    # parser.add_argument("--tasksNum",       default = 20, type = int,       help = "maximal queue size for tasks (to constrain ram usage)") # 2

    parser.add_argument("--useCPU",         action = "store_true",          help = "put word embeddings on cpu")

    # weight loading and training
    parser.add_argument("-r", "--restore",  action = "store_true",          help = "restore last epoch (based on results file)") 
    parser.add_argument("--restoreEpoch",   default = 0, type = int,        help = "if positive, specific epoch to restore")
    parser.add_argument("--weightsToKeep",  default = 2, type = int,        help = "number of previous epochs' weights keep")
    parser.add_argument("--saveEvery",     default = 3000, type = int,      help = "number of iterations to save weights after")
    parser.add_argument("--calleEvery",     default = 1500, type = int,     help = "number of iterations to call custom function after")

    parser.add_argument("--saveSubset",     action = "store_true",          help = "save only subset of the weights")
    parser.add_argument("--trainSubset",    action = "store_true",          help = "train only subset of the weights")
    parser.add_argument("--varSubset",      default = [], nargs = "*",      type = str, help = "list of namespaces to train on")    
    
    # trainReader = ["questionEmbeddings", "questionReader"]
    # saveControl = ["questionEmbeddings", "programEmbeddings", "seqReader", "programControl"]
    
    # experiment files
    parser.add_argument("--expName",        default = "experiment", type = str,    help = "experiment name") 

    # data files
    parser.add_argument("--dataset",         default = "CLEVR", choices = ["CLEVR", "NLVR"], type = str) # 
    parser.add_argument("--dataBasedir",     default = "./", type = str,            help = "data base directory") # /jagupard14/scr1/dorarad/
    parser.add_argument("--generatedPrefix", default = "gennew", type = str,           help = "prefix for generated data files") 
    parser.add_argument("--featureType",     default = "norm_128x32", type = str,   help = "features type") #   
    # resnet101_512x128, norm_400x100, none_80x20, normPerImage_80x20, norm_80x20
    
    ################ optimization

    # training/testing
    parser.add_argument("--train",          action = "store_true",      help = "run training")
    parser.add_argument("--evalTrain",      action = "store_true",      help = "run eval with ema on train dataset") #   
    parser.add_argument("--test",           action = "store_true",      help = "run testing every epoch and generate predictions file") #
    parser.add_argument("--finalTest",      action = "store_true",      help = "run testing on final epoch")
    parser.add_argument("--retainVal",      action = "store_true",      help = "retain validation order between runs") #     

    parser.add_argument("--getPreds",       action = "store_true",      help = "store prediction")
    parser.add_argument("--getAtt",         action = "store_true",      help = "store attention maps")
    parser.add_argument("--analysisType",   default = "", type = str,   choices = ["", "questionLength, programLength","type", "arity"], help = "show breakdown of results according to type") #

    parser.add_argument("--trainedNum",     default = 0, type = int,    help = "if positive, train on subset of the data")    
    parser.add_argument("--testedNum",      default = 0, type = int,    help = "if positive, test on subset of the data")  
    
    # bucketing
    parser.add_argument("--noBucket",       action = "store_true",      help = "bucket data according to question length")        
    parser.add_argument("--noRebucket",     action = "store_true",      help = "bucket data according to question and program length") #
    
    # filtering
    parser.add_argument("--tOnlyChain",     action = "store_true",      help = "train only chain questions")
    parser.add_argument("--vOnlyChain",     action = "store_true",      help = "test only chain questions")
    parser.add_argument("--tMaxQ",          default = 0, type = int,    help = "if positive, train on questions up to this length")
    parser.add_argument("--tMaxP",          default = 0, type = int,    help = "if positive, test on questions up to this length")
    parser.add_argument("--vMaxQ",          default = 0, type = int,    help = "if positive, train on questions with programs up to this length")
    parser.add_argument("--vMaxP",          default = 0, type = int,    help = "if positive, test on questions with programs up to this length")
    parser.add_argument("--tFilterOp",      default = 0, type = int,    help = "train questions by to be included in the types listed")
    parser.add_argument("--vFilterOp",      default = 0, type = int,    help = "test questions by to be included in the types listed")

    # extra and extraVal
    parser.add_argument("--extra",          action = "store_true",      help = "prepare extra data (add to vocabulary") #
    parser.add_argument("--trainExtra",     action = "store_true",      help = "train (only) on extra data") #
    parser.add_argument("--alterExtra",     action = "store_true",      help = "alter main data training with extra dataset") # 
    parser.add_argument("--alterNum",       default = 1, type = int,    help = "alteration rate") #
    parser.add_argument("--extraVal",       action = "store_true",      help = "only extra validation data (for compositional clevr)") # 
    parser.add_argument("--finetuneNum",    default = 0, type = int,    help = "if positive, finetune on that subset of val (for compositional clevr)") #

    # exponential moving average
    parser.add_argument("--useEMA",         action = "store_true",           help = "use exponential moving average for weights")
    parser.add_argument("--emaDecayRate",   default = 0.999, type = float,   help = "decay rate for exponential moving average")
    
    # sgd optimizer
    parser.add_argument("--batchSize",      default = 64, type = int,       help = "batch size")    
    parser.add_argument("--epochs",         default = 100, type = int,      help = "number of epochs to run")    
    parser.add_argument("--lr",             default = 0.0001, type = float, help = "learning rate")
    parser.add_argument("--lrReduce",       action = "store_true",          help = "reduce learning rate if training loss doesn't go down (manual annealing)")    
    parser.add_argument("--lrDecayRate",    default = 0.5, type = float,    help = "learning decay rate if training loss doesn't go down")
    parser.add_argument("--earlyStopping",  default = 0, type = int,        help = "if positive, stop if no improvement for that number of epochs")

    parser.add_argument("--adam",           action = "store_true",          help = "use adam")   
    parser.add_argument("--l2",             default = 0, type = float,      help = "if positive, add l2 loss term")    
    parser.add_argument("--clipGradients",  action = "store_true",          help = "clip gradients")
    parser.add_argument("--gradMaxNorm",    default = 8, type = int,        help = "clipping value") 

    # batch normalization
    parser.add_argument("--memoryBN",   action = "store_true",              help = "use batch normalization on the recurrent memory")
    parser.add_argument("--stemBN",     action = "store_true",              help = "use batch normalization in the image input unit (stem)")
    parser.add_argument("--outputBN",   action = "store_true",              help = "use batch normalization in the output unit")
    parser.add_argument("--bnDecay",    default = 0.999, type = float,      help = "batch norm decay rate")
    parser.add_argument("--bnCenter",   action = "store_true",              help = "batch norm with centering")
    parser.add_argument("--bnScale",    action = "store_true",              help = "batch norm with scaling")

    ## dropouts
    parser.add_argument("--encInputDropout", default = 0.85, type = float,   help = "dropout of the rnn inputs to the Question Input Unit") 
    parser.add_argument("--encStateDropout", default = 1.0, type = float,   help = "dropout of the rnn states of the Question Input Unit") 
    parser.add_argument("--stemDropout",     default = 0.82, type = float,  help = "dropout of the Image Input Unit (the stem)")

    parser.add_argument("--qDropout",       default = 0.92, type = float,    help = "dropout on the question vector") 
    # parser.add_argument("--qDropoutOut",    default = 1.0, type = float,    help = "dropout on the question vector the goes to the output unit") 
    # parser.add_argument("--qDropoutMAC",    default = 1.0, type = float,    help = "dropout on the question vector the goes to MAC") 

    parser.add_argument("--memoryDropout",  default = 0.85, type = float,    help = "dropout on the recurrent memory") 
    parser.add_argument("--readDropout",    default = 0.85, type = float,    help = "dropout of the read unit")     
    parser.add_argument("--writeDropout",   default = 1.0, type = float,    help = "dropout of the write unit") 
    parser.add_argument("--outputDropout",  default = 0.85, type = float,   help = "dropout of the output unit") 
    
    parser.add_argument("--parametricDropout",        action = "store_true", help = "use parametric dropout") #
    parser.add_argument("--encVariationalDropout",    action = "store_true", help = "use variational dropout in the RNN input unit") 
    parser.add_argument("--memoryVariationalDropout", action = "store_true", help = "use variational dropout across the MAC network") 

    ## nonlinearities
    parser.add_argument("--relu",       default = "STD", choices = ["STD", "PRM", "ELU", "LKY", "SELU"], type = str, help = "type of ReLU to use: standard, parametric, ELU, or leaky")
    # parser.add_argument("--reluAlpha",  default = 0.2, type = float,    help = "alpha value for the leaky ReLU")

    parser.add_argument("--mulBias",    default = 0.0, type = float,   help = "bias to add in multiplications (x + b) * (y + b) for better training") #

    parser.add_argument("--imageLinPool",   default = 2, type = int,   help = "pooling for image linearizion") 

    ################ baseline model parameters
    
    parser.add_argument("--useBaseline",    action = "store_true",     help = "run the baseline model")    
    parser.add_argument("--baselineLSTM",   action = "store_true",     help = "use LSTM in baseline")    
    parser.add_argument("--baselineCNN",    action = "store_true",     help = "use CNN in baseline")       
    parser.add_argument("--baselineAtt",    action = "store_true",     help = "use stacked attention baseline")
    
    parser.add_argument("--baselineProjDim", default = 64, type = int, help = "projection dimension for image linearizion")    

    parser.add_argument("--baselineAttNumLayers", default = 2, type = int, help = "number of stacked attention layers") 
    parser.add_argument("--baselineAttType", default = "ADD", type = str, choices = ["MUL", "DIAG", "BL", "ADD"], help = "attention type (multiplicative, additive, etc)") 

    ################ image input unit (the "stem")

    parser.add_argument("--stemDim",         default = 512, type = int,               help = "dimension of stem CNNs") 
    parser.add_argument("--stemNumLayers",   default = 2, type = int,                 help = "number of stem layers")
    parser.add_argument("--stemKernelSize",  default = 3, type = int,                 help = "kernel size for stem (same for all the stem layers)")
    parser.add_argument("--stemKernelSizes", default = None, nargs = "*", type = int, help = "kernel sizes for stem (per layer)")
    parser.add_argument("--stemStrideSizes", default = None, nargs = "*", type = int, help = "stride sizes for stem (per layer)")

    parser.add_argument("--stemLinear",             action = "store_true",          help = "use a linear stem (instead of CNNs)") #
    # parser.add_argument("--stemProjDim",          default = 64, type = int,       help = "projection dimension of in image linearization") #
    # parser.add_argument("--stemProjPooling",      default = 2, type = int,        help = "pooling for the image linearization") #

    parser.add_argument("--stemGridRnn",            action = "store_true",          help = "use grid RNN layer") #
    parser.add_argument("--stemGridRnnMod",         default = "RNN", type = str,    choices = ["RNN", "GRU"], help = "RNN type for grid") #
    parser.add_argument("--stemGridAct",            default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "nonlinearity type for grid") #

    ## location
    parser.add_argument("--locationAware",          action = "store_true",          help = "add positional features to image representation (linear meshgrid by default)") 
    parser.add_argument("--locationType",           default = "L", type = str,      choices = ["L", "PE"], help = "L: linear features, PE: Positional Encoding") 
    parser.add_argument("--locationBias",           default = 1.0, type = float,    help = "the scale of the positional features")
    parser.add_argument("--locationDim",            default = 32, type = int,       help = "the number of PE dimensions")

    ################ question input unit (the "encoder")
    parser.add_argument("--encType",                default = "LSTM",               choices = ["RNN", "GRU", "LSTM", "MiGRU", "MiLSTM"], help = "encoder RNN type")
    parser.add_argument("--encDim",                 default = 512, type = int,      help = "dimension of encoder RNN")    
    parser.add_argument("--encNumLayers",           default = 1, type = int,        help = "number of encoder RNN layers")  
    parser.add_argument("--encBi",                  action = "store_true",          help = "use bi-directional encoder")    
    # parser.add_argument("--encOutProj",           action = "store_true",          help = "add projection layer for encoder outputs") 
    # parser.add_argument("--encOutProjDim",        default = 256, type = int,      help = "dimension of the encoder projection layer") 
    # parser.add_argument("--encQProj",             action = "store_true",          help = "add projection for the question representation")
    parser.add_argument("--encProj",                action = "store_true",          help = "project encoder outputs and question")
    parser.add_argument("--encProjQAct",            default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "project question vector with this activation")

    ##### word embeddings 
    parser.add_argument("--wrdEmbDim",              default = 300, type = int,      help = "word embeddings dimension") 
    parser.add_argument("--wrdEmbRandom",           action = "store_true",          help = "initialize word embeddings to random (normal)")
    parser.add_argument("--wrdEmbUniform",          action = "store_true",          help = "initialize with uniform distribution")
    parser.add_argument("--wrdEmbScale",            default = 1.0, type = float,    help = "word embeddings initialization scale")
    parser.add_argument("--wrdEmbFixed",            action = "store_true",          help = "set word embeddings fixed (don't train)")
    parser.add_argument("--wrdEmbUnknown",          action = "store_true",          help = "set words outside of training set to <UNK>")

    parser.add_argument("--ansEmbMod",              default = "NON", choices = ["NON", "SHARED", "BOTH"], type = str,   help = "BOTH: create word embeddings for answers. SHARED: share them with question embeddings.") #
    parser.add_argument("--answerMod",              default = "NON", choices = ["NON", "MUL", "DIAG", "BL"], type = str, help = "operation for multiplication with answer embeddings: direct multiplication, scalar weighting, or bilinear") #

    ################ output unit (classifier)
    parser.add_argument("--outClassifierDims",      default = [512], nargs = "*",   type = int, help = "dimensions of the classifier") 
    parser.add_argument("--outImage",               action = "store_true",          help = "feed the image to the output unit") 
    parser.add_argument("--outImageDim",            default = 1024, type = int,     help = "dimension of linearized image fed to the output unit") 
    parser.add_argument("--outQuestion",            action = "store_true",          help = "feed the question to the output unit") 
    parser.add_argument("--outQuestionMul",         action = "store_true",          help = "feed the multiplication of question and memory to the output unit") 

    ################ network
    
    parser.add_argument("--netLength",              default = 16, type = int,        help = "network length (number of cells)")      
    # parser.add_argument("--netDim", default = 512, type = int)
    parser.add_argument("--memDim",                 default = 512, type = int,      help = "dimension of memory state")
    parser.add_argument("--writeDim",               default=272, type=int,          help="dimension of capsule memory state")
    parser.add_argument("--ctrlDim",                default = 512, type = int,      help = "dimension of control state")
    parser.add_argument("--attDim",                 default = 512, type = int,      help = "dimension of pre-attention interactions space")
    parser.add_argument("--unsharedCells",          default = False, type = bool,   help = "unshare weights between cells ") 

    # initialization
    parser.add_argument("--initCtrl",               default = "PRM", type = str,    choices = ["PRM", "ZERO", "Q"], help = "initialization mod for control")
    parser.add_argument("--initMem",                default = "PRM", type = str,    choices = ["PRM", "ZERO", "Q"], help = "initialization mod for memory")
    parser.add_argument("--initKBwithQ",            default = "NON", type = str,    choices = ["NON", "CNCT", "MUL"], help = "merge question with knowledge base")
    parser.add_argument("--addNullWord",            action = "store_true",          help = "add parametric word in the beginning of the question") 

    ################ control unit
    # control ablations (use whole question or pre-attention continuous vectors as control)
    parser.add_argument("--controlWholeQ",          action = "store_true",          help = "use whole question vector as control") 
    parser.add_argument("--controlContinuous",      action = "store_true",          help = "use continuous representation of control (without attention)")

    # step 0: inputs to control unit (word embeddings or encoder outputs, with optional projection)  
    parser.add_argument("--controlContextual",      action = "store_true",          help = "use contextual words for attention (otherwise will use word embeddings)")
    parser.add_argument("--controlInWordsProj",     action = "store_true",          help = "apply linear projection over words for attention computation") 
    parser.add_argument("--controlOutWordsProj",    action = "store_true",          help = "apply linear projection over words for summary computation") 

    parser.add_argument("--controlInputUnshared",   action = "store_true",          help = "use different question representation for each cell") 
    parser.add_argument("--controlInputAct",        default = "TANH", type = str,   choices = ["NON", "RELU", "TANH"], help = "activation for question projection")

    # step 1: merging previous control and whole question  
    parser.add_argument("--controlFeedPrev",        action = "store_true",          help = "feed previous control state") 
    parser.add_argument("--controlFeedPrevAtt",     action = "store_true",          help = "feed previous control post word attention (otherwise will feed continuous control)")
    parser.add_argument("--controlFeedInputs",      action = "store_true",          help = "feed question representation")
    parser.add_argument("--controlContAct",         default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "activation on the words interactions")
    
    # step 2: word attention and optional projection 
    parser.add_argument("--controlConcatWords",     action = "store_true",          help = "concatenate words to interaction when computing attention") 
    parser.add_argument("--controlProj",            action = "store_true",          help = "apply linear projection on words interactions")
    parser.add_argument("--controlProjAct",         default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "activation for control interactions")

    # parser.add_argument("--controlSelfAtt", default = False, type = bool) 

    # parser.add_argument("--controlCoverage", default = False, type = bool)
    # parser.add_argument("--controlCoverageBias", default = 1.0, type = float)

    # parser.add_argument("--controlPostRNN", default = False, type = bool) 
    # parser.add_argument("--controlPostRNNmod", default = "RNN", type = str) # GRU

    # parser.add_argument("--selfAttShareInter", default = False, type = bool)

    # parser.add_argument("--wordControl", default = False, type = bool)
    # parser.add_argument("--gradualControl", default = False, type = bool)

    ################ capsules
    parser.add_argument("--NUM_VIS_CAPS_L1", type=int, default=16, help="num. of prime capsules")
    parser.add_argument("--NUM_VIS_CAPS_L2", type=int, default=16, help="num. of second layer capsules")
    parser.add_argument("--inv_temp_delta", type=float, default=0.1, help="inv_temp_delta")
    parser.add_argument("--inv_temp", type=float, default=0.5, help="inv_temp")
    parser.add_argument("--combineMask", action="store_true", help="combine prev memory and control to generate mask")


    ################ read unit
    # step 1: KB-memory interactions
    parser.add_argument("--readProjInputs",         action = "store_true",         help = "project read unit inputs")
    parser.add_argument("--readProjShared",         action = "store_true",         help = "use shared projection for all read unit inputs")

    parser.add_argument("--readMemAttType",         default = "MUL", type = str,   choices = ["MUL", "DIAG", "BL", "ADD"], help = "attention type for interaction with memory")
    parser.add_argument("--readMemConcatKB",        action = "store_true",         help = "concatenate KB elements to memory interaction")
    parser.add_argument("--readMemConcatProj",      action = "store_true",         help = "concatenate projected values instead or original to memory interaction")
    parser.add_argument("--readMemProj",            action = "store_true",         help = "project interactions with memory")
    parser.add_argument("--readMemAct",             default = "RELU", type = str,  choices = ["NON", "RELU", "TANH"], help = "activation for memory interaction")

    # step 2: interaction with control
    parser.add_argument("--readCtrl",               action = "store_true",         help = "compare KB-memory interactions to control")
    parser.add_argument("--readCtrlAttType",        default = "MUL", type = str,   choices = ["MUL", "DIAG", "BL", "ADD"], help = "attention type for interaction with control")
    parser.add_argument("--readCtrlConcatKB",       action = "store_true",         help = "concatenate KB elements to control interaction")
    parser.add_argument("--readCtrlConcatProj",     action = "store_true",         help = "concatenate projected values instead or original to control interaction")
    parser.add_argument("--readCtrlConcatInter",    action = "store_true",         help = "concatenate memory interactions to control interactions")
    parser.add_argument("--readCtrlAct",            default = "RELU", type = str,  choices = ["NON", "RELU", "TANH"], help = "activation for control interaction")

    # step 3: summarize attention over knowledge base
    parser.add_argument("--readSmryKBProj",       action = "store_true",        help = "use knowledge base projections when summing attention up (should be used only if KB is projected.")
    
    # parser.add_argument("--saAllMultiplicative", default = False, type = bool) 
    # parser.add_argument("--saSumMultiplicative", default = False, type = bool)

    ################ write unit
    # step 1: input to the write unit (only previous memory, or new information, or both)
    parser.add_argument("--writeInputs",            default = "BOTH", type = str,   choices = ["MEM", "INFO", "BOTH", "SUM"], help = "inputs to the write unit")
    parser.add_argument("--writeConcatMul",         action = "store_true",          help = "add multiplicative integration between inputs")
    
    parser.add_argument("--writeInfoProj",          action = "store_true",          help = "project retrieved info")
    parser.add_argument("--writeInfoAct",           default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "new info activation")

    # step 2: self attention and following projection
    parser.add_argument("--writeSelfAtt",           action = "store_true",          help = "use self attention") 
    parser.add_argument("--writeSelfAttMod",        default = "NON", type = str,    choices = ["NON", "CONT"], help = "control version to compare to")

    parser.add_argument("--writeMergeCtrl",           action = "store_true",          help = "merge control with memory") 

    parser.add_argument("--writeMemProj",           action = "store_true",          help = "project new memory")
    parser.add_argument("--writeMemAct",            default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "new memory activation")

    # step 3: gate between new memory and previous value 
    parser.add_argument("--writeGate",              action = "store_true",          help = "add gate to write unit") 
    parser.add_argument("--writeGateShared",        action = "store_true",          help = "use one gate value for all dimensions of the memory state") 
    parser.add_argument("--writeGateBias",          default = 1.0, type = float,    help = "bias for the write unit gate (positive to bias for taking new memory)") 

    ## modular
    # parser.add_argument("--modulesNum", default = 10, type = int) 
    # parser.add_argument("--controlBoth", default = False, type = bool)
    # parser.add_argument("--addZeroModule", default = False, type = bool) 
    # parser.add_argument("--endModule", default = False, type = bool) 

    ## hybrid
    # parser.add_argument("--hybrid",      default = False, type = bool, help = "hybrid attention cnn model") 
    # parser.add_argument("--earlyHybrid", default = False, type = bool) 
    # parser.add_argument("--lateHybrid",  default = False, type = bool) 

    ## autoencoders
    # parser.add_argument("--autoEncMem",         action = "store_true",          help = "add memory2control auto-encoder loss")
    # parser.add_argument("--autoEncMemW",        default = 0.0001, type = float, help = "weight for auto-encoder loss")
    # parser.add_argument("--autoEncMemInputs",   default = "INFO", type = str,   choices = ["MEM", "INFO"], help = "inputs to auto-encoder")
    # parser.add_argument("--autoEncMemAct",      default = "NON", type = str,    choices = ["NON", "RELU", "TANH"], help = "activation type in the auto-encoder")
    # parser.add_argument("--autoEncMemLoss",     default = "CONT", type = str,   choices = ["CONT", "PROB", "SMRY"], help = "target for the auto-encoder loss")
    # parser.add_argument("--autoEncMemCnct",     action = "store_true",          help = "concat word attentions to auto-encoder features")

    # parser.add_argument("--autoEncCtrl",        action = "store_true")
    # parser.add_argument("--autoEncCtrlW",       default = 0.0001, type = float)
    # parser.add_argument("--autoEncCtrlGRU",     action = "store_true")
 
    ## temperature
    # parser.add_argument("--temperature",    default = 1.0, type = float,        help = "temperature for modules softmax") #
    # parser.add_argument("--tempParametric", action = "store_true",              help = "parametric temperature") #
    # parser.add_argument("--tempDynamic",    action = "store_true",              help = "dynamic temperature") #
    # parser.add_argument("--tempAnnealRate", default = 0.000004, type = float,   help = "temperature annealing rate") #
    # parser.add_argument("--tempMin",        default = 0.5, type = float,        help = "minimum temperature") #

    ## gumbel
    # parser.add_argument("--gumbelSoftmax",      action = "store_true", help = "use gumbel for the module softmax (soft for training and hard for testing)") #
    # parser.add_argument("--gumbelSoftmaxBoth",  action = "store_true", help = "use softmax for training and testing") #
    # parser.add_argument("--gumbelArgmaxBoth",   action = "store_true", help = "use argmax for training and testing") #
    
    parser.parse_args(namespace = config) 

###################################### dataset configuration ######################################

def configCLEVR():
    config.dataPath = "{dataBasedir}/CLEVR_v1/data".format(dataBasedir = config.dataBasedir)
    config.datasetFilename = "CLEVR_{tier}_questions.json"
    config.wordVectorsFile = "./CLEVR_v1/data/glove/glove.6B.{dim}d.txt".format(dim = config.wrdEmbDim) #

    config.imageDims = [14, 14, 1024]
    config.programLims = [5, 10, 15, 20]
    config.questionLims = [10, 15, 20, 25]        

def configNLVR():
    config.dataPath = "{dataBasedir}/nlvr".format(dataBasedir = config.dataBasedir)
    config.datasetFilename = "{tier}.json"
    config.imagesFilename = "{{tier}}_{featureType}.h5".format(featureType = config.featureType)
    config.imgIdsFilename = "{tier}ImgIds.json"
    config.wordVectorsFile = "./CLEVR_v1/data/glove/glove.6B.{dim}d.txt".format(dim = config.wrdEmbDim) #

    config.questionLims = [12]
    # config.noRebucket = True 

    # if config.stemKernelSizes == []:
    #     if config.featureType.endsWith("128x32"):
    #         config.stemKernelSizes = [8, 4, 4]
    #         config.stemStrideSizes = [2, 2, 1]
    #         config.stemNumLayers = 3
    #     if config.featureType.endsWith("512x128"): 
    #         config.stemKernelSizes = [8, 4, 4, 2]
    #         config.stemStrideSizes = [4, 2, 2, 1]
    #         config.stemNumLayers = 4
    # config.stemDim = 64

    if config.featureType == "resnet101_512x128":
        config.imageDims = [8, 32, 1024]
    else:
        stridesOverall = 1
        if stemStrideSizes is not None:
            for s in config.stemStrideSizes:
                stridesOverall *= int(s)
        size = config.featureType.split("_")[-1].split("x")
        config.imageDims = [int(size[1]) / stridesOverall, int(size[0]) / stridesOverall, 3]

## dataset specific configs
loadDatasetConfig = {
    "CLEVR": configCLEVR,
    "NLVR": configNLVR
}
