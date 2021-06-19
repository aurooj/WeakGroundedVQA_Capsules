import collections
import numpy as np
import tensorflow as tf
from tensorflow import newaxis as ax

import ops
from config import config
from capsules.capsules import reshape_vis_caps_spatial2, create_prim_conv2d_caps, create_conv2d_caps

MACCellTuple = collections.namedtuple("MACCellTuple", ("control", "memory"))

'''
The MAC cell.

Recurrent cell for multi-step reasoning. Presented in https://arxiv.org/abs/1803.03067.
The cell has recurrent control and memory states that interact with the question 
and knowledge base (image) respectively.

The hidden state structure is MACCellTuple(control, memory)

At each step the cell performs by calling to three subunits: control, read and write.

1. The Control Unit computes the control state by computing attention over the question words.
The control state represents the current reasoning operation the cell performs.

2. The Read Unit retrieves information from the knowledge base, given the control and previous
memory values, by computing 2-stages attention over the knowledge base.

3. The Write Unit integrates the retrieved information to the previous hidden memory state,
given the value of the control state, to perform the current reasoning operation.
'''
class MACCell(tf.nn.rnn_cell.RNNCell):

    '''Initialize the MAC cell. 
    (Note that in the current version the cell is stateful -- 
    updating its own internals when being called) 
    
    Args:
        vecQuestions: the vector representation of the questions. 
        [batchSize, ctrlDim]

        questionWords: the question words embeddings. 
        [batchSize, questionLength, ctrlDim]

        questionCntxWords: the encoder outputs -- the "contextual" question words. 
        [batchSize, questionLength, ctrlDim]

        questionLengths: the length of each question.
        [batchSize]
        
        memoryDropout: dropout on the memory state (Tensor scalar).
        readDropout: dropout inside the read unit (Tensor scalar).
        writeDropout: dropout on the new information that gets into the write unit (Tensor scalar).
        
        batchSize: batch size (Tensor scalar).
        train: train or test mod (Tensor boolean).
        reuse: reuse cell

        knowledgeBase: 
    '''
    def __init__(self, vecQuestions, questionWords, questionCntxWords, questionLengths, 
            knowledgeBase, memoryDropout, readDropout, 
            writeDropout, controlDropoutPre, controlDropoutPost, wordDropout, vocabDropout,
            objectDropout, batchSize, train, kbSize = None, reuse = None):
        
        self.vecQuestions = vecQuestions
        self.questionWords = questionWords
        self.questionCntxWords = questionCntxWords
        self.questionLengths = questionLengths
        knowledgeBase = tf.concat(knowledgeBase, axis=-1)
        #print(knowledgeBase.get_shape())
        self.knowledgeBase = knowledgeBase
        self.kbSize = kbSize

        self.dropouts = {}
        self.dropouts["memory"] = memoryDropout 
        self.dropouts["read"] = readDropout 
        self.dropouts["write"] = writeDropout
        self.dropouts["controlPre"] = controlDropoutPre
        self.dropouts["controlPost"] = controlDropoutPost
        self.dropouts["word"] = wordDropout
        self.dropouts["vocab"] = vocabDropout
        self.dropouts["object"] = objectDropout

        self.none = tf.zeros((batchSize, 1), dtype = tf.float32)

        self.batchSize = batchSize
        self.train = train
        self.reuse = reuse

    ''' 
    Cell state size. 
    '''
    @property
    def state_size(self):
        return MACCellTuple(config.ctrlDim, config.writeDim)

    '''
    Cell output size. Currently it doesn't have any outputs. 
    '''
    @property
    def output_size(self):
        return 1

    '''
    The Control Unit: computes the new control state -- the reasoning operation,
    by summing up the word embeddings according to a computed attention distribution.
    
    The unit is recurrent: it receives the whole question and the previous control state,
    merge them together (resulting in the "continuous control"), and then uses that 
    to compute attentions over the question words. Finally, it combines the words 
    together according to the attention distribution to get the new control state. 
    
    Args:
        controlInput: external inputs to control unit (the question vector).
        [batchSize, ctrlDim]

        inWords: the representation of the words used to compute the attention.
        [batchSize, questionLength, ctrlDim]

        outWords: the representation of the words that are summed up. 
                  (by default inWords == outWords)
        [batchSize, questionLength, ctrlDim]

        questionLengths: the length of each question.
        [batchSize]

        control: the previous control hidden state value.
        [batchSize, ctrlDim]

        contControl: optional corresponding continuous control state
        (before casting the attention over the words).
        [batchSize, ctrlDim]

    Returns:
        the new control state
        [batchSize, ctrlDim]

        the continuous (pre-attention) control
        [batchSize, ctrlDim]
    '''
    def control(self, controlInput, inWords, outWords, questionLengths,
        append = True, control = None, contControl = None, name = "", reuse = None):

        with tf.variable_scope("control" + name, reuse = reuse):
            dim = config.ctrlDim

            ## Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = controlInput
            if config.controlFeedPrev:
                newContControl = control if config.controlFeedPrevAtt else contControl
                if config.controlFeedInputs:
                    newContControl = tf.concat([newContControl, controlInput], axis = -1)
                    dim += config.ctrlDim

                # merge inputs together
                newContControl = ops.linear(newContControl, dim, config.ctrlDim,
                    act = config.controlContAct, name = "contControl")
                dim = config.ctrlDim

            ## Step 2: compute attention distribution over words and sum them up accordingly.
            # compute interactions with question words
            interactions = tf.expand_dims(newContControl, axis = 1) * inWords
            
            # optionally concatenate words
            if config.controlConcatWords:
                interactions = tf.concat([interactions, inWords], axis = -1)
                dim += config.ctrlDim                                              

            # optional projection
            if config.controlProj:
                interactions = ops.linear(interactions, dim, config.ctrlDim, 
                    act = config.controlProjAct) 
                dim = config.ctrlDim

            # compute attention distribution over words and summarize them accordingly 
            logits = ops.inter2logits(interactions, dim)

            if config.wordDp < 1.0:
                logits = tf.nn.dropout(logits, self.dropouts["word"])

            attention = tf.nn.softmax(ops.expMask(logits, questionLengths))
            if append:
                self.attentions["question"].append(attention)

            newControl = ops.att2Smry(attention, outWords) 
            
            gateDim = config.ctrlDim

            # ablation: use continuous control (pre-attention) instead
            if config.controlContinuous:
                newControl = newContControl

        return newControl, newContControl

    '''
    The read unit extracts relevant information from the knowledge base given the
    cell's memory and control states. It computes attention distribution over
    the knowledge base by comparing it first to the memory and then to the control.
    Finally, it uses the attention distribution to sum up the knowledge base accordingly,
    resulting in an extraction of relevant information. 

    Args:
        knowledge base: representation of the knowledge base (image). 
        [batchSize, kbSize (Height * Width), memDim]

        memory: the cell's memory state
        [batchSize, memDim]

        control: the cell's control state
        [batchSize, ctrlDim]

    Returns the information extracted.
    [batchSize, memDim]
    '''
    def read(self, knowledgeBase, memory, control, name = "", reuse = None):
        with tf.variable_scope("read" + name, reuse = reuse):
            dim = config.memDim
            interDim = dim
            ## mask
            knowledgeBase = self.queryFocusedSoftmasking(control, dim, knowledgeBase)

            ## memory dropout
            if config.memoryVariationalDropout:
                memory = ops.applyVarDpMask(memory, self.memDpMask, self.dropouts["memory"])
            else:
                memory = tf.nn.dropout(memory, self.dropouts["memory"])

            ## Step 1: knowledge base / memory interactions 
            # parameters for knowledge base and memory projection 
            proj = None
            if config.readProjInputs:
                proj = {"dim": config.attDim, "shared": config.readProjShared, "dropout": self.dropouts["read"] }
                dim = config.attDim

            # parameters for concatenating knowledge base elements
            concat = {"x": config.readMemConcatKB, "proj": config.readMemConcatProj}

            # compute interactions between knowledge base and memory
            interactions, interDim = ops.mul(x = knowledgeBase, y = memory, dim = config.writeDim,
                proj = proj, concat = concat, interMod = config.readMemAttType, name = "memInter")

            projectedKB = proj.get("x") if proj else None

            # project memory interactions back to hidden dimension
            if config.readMemProj:
                interactions = ops.linear(interactions, interDim, dim, act = config.readMemAct, 
                    name = "memKbProj")
            else:
                dim = interDim

            ## Step 2: compute interactions with control
            if config.readCtrl:
                # compute interactions with control
                if config.ctrlDim != dim:
                    control = ops.linear(control, config.ctrlDim, dim, name = "ctrlProj")

                interactions, interDim = ops.mul(interactions, control, dim, 
                    interMod = config.readCtrlAttType, concat = {"x": config.readCtrlConcatInter}, 
                    name = "ctrlInter")

                # optionally concatenate knowledge base elements
                if config.readCtrlConcatKB:
                    if config.readCtrlConcatProj:
                        addedInp, addedDim = projectedKB, config.attDim
                    else:
                        addedInp, addedDim = knowledgeBase, config.writeDim
                    interactions = tf.concat([interactions, addedInp], axis = -1)
                    dim += addedDim   

                # optional nonlinearity 
                interactions = ops.activations[config.readCtrlAct](interactions)

            ## Step 3: sum attentions up over the knowledge base
            # transform vectors to attention distribution
            if config.objectDp < 1.0:
                interactions = tf.nn.dropout(interactions, self.dropouts["object"])

            attention = ops.inter2att(interactions, dim, dropout = self.dropouts["read"], 
                mask = self.kbSize)

            self.attentions["kb"].append(attention)

            # optionally use projected knowledge base instead of original
            if config.readSmryKBProj:
                knowledgeBase = projectedKB
            
            # sum up the knowledge base according to the distribution
            information = ops.att2Smry(attention, knowledgeBase)

        return information

    def queryFocusedSoftmasking(self, control, dim, knowledgeBase):
        #this function generates a control (query) based soft mask and returns masked capsules as output.
        #knowledgeBase is visual capsules here
        _, d = control.get_shape().as_list()
        control_ = ops.linear(control, inDim=d, outDim=dim,
                              name='control_linear')  # out dim is dim based on capsules
        #mask generation
        mask_logits = ops.linear(control_, inDim=dim, outDim=config.NUM_VIS_CAPS_L2,
                                 name='mask_linear')  # position aware mask
        mask = tf.nn.softmax(mask_logits, axis=-1)

        # print('inside read...')
        # print(knowledgeBase.shape)
        # print(mask.shape)

        #apply mask to capsules
        knowledgeBase = tf.concat(knowledgeBase, axis=-1)
        attnknowledgeBase = knowledgeBase * mask[:, ax, ax, :, ax]

        #reshaping visual capsules
        knowledgeBase = reshape_vis_caps_spatial2(attnknowledgeBase)  # B,H,W, C*17
        return knowledgeBase

    '''
    The write unit integrates newly retrieved information (from the read unit),
    with the cell's previous memory hidden state, resulting in a new memory value.
    The unit optionally supports:
    1. Self-attention to previous control / memory states, in order to consider previous steps
    in the reasoning process.
    2. Gating between the new memory and previous memory states, to allow dynamic adjustment
    of the reasoning process length.

    Args:
        memory: the cell's memory state
        [batchSize, memDim]

        info: the information to integrate with the memory
        [batchSize, memDim]

        control: the cell's control state
        [batchSize, ctrlDim]

        contControl: optional corresponding continuous control state 
        (before casting the attention over the words).
        [batchSize, ctrlDim]

    Return the new memory 
    [batchSize, memDim]
    '''
    def write(self, memory, info, control, contControl = None, name = "", reuse = None):
        with tf.variable_scope("write" + name, reuse = reuse):

            # optionally project info
            if config.writeInfoProj:
                info = ops.linear(info, config.writeDim, config.writeDim, name = "info")

            # optional info nonlinearity
            info = ops.activations[config.writeInfoAct](info) 

            # compute self-attention vector based on previous controls and memories
            if config.writeSelfAtt:
                selfControl = control
                if config.writeSelfAttMod == "CONT":
                    selfControl = contControl
                selfControl = ops.linear(selfControl, config.ctrlDim, config.ctrlDim, name = "ctrlProj")
                
                interactions = self.controls * tf.expand_dims(selfControl, axis = 1)

                attention = ops.inter2att(interactions, config.ctrlDim, name = "selfAttention")
                self.attentions["self"].append(attention) 
                selfSmry = ops.att2Smry(attention, self.memories)

            # get write unit inputs: previous memory, the new info, optionally self-attention / control
            newMemory, dim = memory, config.writeDim
            if config.writeInputs == "INFO":
                newMemory = info
            elif config.writeInputs == "SUM":
                newMemory += info
            elif config.writeInputs == "BOTH":
                newMemory, dim = ops.concat(newMemory, info, dim, mul = config.writeConcatMul)
            # else: MEM

            if config.writeSelfAtt:
                newMemory = tf.concat([newMemory, selfSmry], axis = -1)
                dim += config.writeDim

            if config.writeMergeCtrl:
                #project memory back to memory dimension
                if config.writeMemProj or (dim != config.writeDim):
                    newMemory = ops.linear(newMemory, dim, config.writeDim, name = "newMemoryCtrl")
                    dim = config.writeDim
                    
                if config.writeMergeCtrlMul:
                    newMemory = tf.concat([newMemory, newMemory * control], axis = -1) # control, 
                    dim += config.writeDim # 2 *
                else:
                    newMemory = tf.concat([newMemory, control], axis = -1)
                    dim += config.writeDim

            # project memory back to memory dimension
            if config.writeMemProj or (dim != config.writeDim):
                newMemory = ops.linear(newMemory, dim, config.writeDim, name = "newMemory")

            # optional memory nonlinearity
            newMemory = ops.activations[config.writeMemAct](newMemory)

            # write unit gate
            if config.writeGate:
                gateDim = config.writeDim
                if config.writeGateShared:
                    gateDim = 1
                
                z = tf.sigmoid(ops.linear(control, config.ctrlDim, gateDim, name = "gate", bias = config.writeGateBias))
                
                self.attentions["gate"].append(z)

                newMemory = newMemory * z + memory * (1 - z)                

            # optional batch normalization
            if config.memoryBN:
                newMemory = tf.contrib.layers.batch_norm(newMemory, decay = config.bnDecay, 
                    center = config.bnCenter, scale = config.bnScale, 
                    is_training = self.train, updates_collections = None)

        return newMemory

    def memAutoEnc(newMemory, info, control, name = "", reuse = None):
        with tf.variable_scope("memAutoEnc" + name, reuse = reuse):
            # inputs to auto encoder
            features = info if config.autoEncMemInputs == "INFO" else newMemory
            features = ops.linear(features, config.memDim, config.ctrlDim, 
                act = config.autoEncMemAct, name = "aeMem")

            # reconstruct control
            if config.autoEncMemLoss == "CONT":
                loss = tf.reduce_mean(tf.squared_difference(control, features))
            else:                    
                interactions, dim = ops.mul(self.questionCntxWords, features, config.ctrlDim, 
                    concat = {"x": config.autoEncMemCnct}, mulBias = config.mulBias, name = "aeMem")
                
                logits = ops.inter2logits(interactions, dim)
                logits = ops.expMask(logits, self.questionLengths)

                # reconstruct word attentions
                if config.autoEncMemLoss == "PROB":
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels = self.attentions["question"][-1], logits = logits))
                
                # reconstruct control through words attentions
                else:
                    attention = tf.nn.softmax(logits)
                    summary = ops.att2Smry(attention, self.questionCntxWords)
                    loss = tf.reduce_mean(tf.squared_difference(control, summary))
        
        return loss

    '''
    Call the cell to get new control and memory states.

    Args:
        inputs: in the current implementation the cell don't get recurrent inputs
        every iteration (argument for comparability with rnn interface).
            
        state: the cell current state (control, memory)
        MACCellTuple([batchSize, ctrlDim],[batchSize, memDim])

    Returns the new state -- the new memory and control values.
    MACCellTuple([batchSize, ctrlDim],[batchSize, memDim])
    '''
    def __call__(self, inputs, state, scope = None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope, reuse = self.reuse): #  as tfscope
            control = state.control
            memory = state.memory

            # cell sharing
            inputName = "qInput"
            inputNameU = "qInputU"
            inputReuseU = inputReuse = (self.iteration > 0)
            if config.controlInputUnshared:
                inputNameU = "qInput%d" % self.iteration
                inputReuseU = None

            cellName = ""
            cellReuse = (self.iteration > 0)
            if config.unsharedCells:
                cellName = str(self.iteration)
                cellReuse = None 

            ## control unit
            # prepare question input to control 
            controlInput = ops.linear(self.vecQuestions, config.ctrlDim, config.ctrlDim, 
                name = inputName, reuse = inputReuse)

            if not config.linearControl:
                controlInput = ops.activations[config.controlInputAct](controlInput)

                controlInput = ops.linear(controlInput, config.ctrlDim, config.ctrlDim, 
                    name = inputNameU, reuse = inputReuseU)

            if config.controlInWordsProj and config.controlOutWordsProj:
                if config.controlPostDropout < 1.0:
                    inWords = tf.nn.dropout(self.inWords, self.dropouts["controlPost"])
                    outWords = inWords
                else:
                    inWords = self.inWords
                    outWords = self.outWords                    
            else:
                inWords = self.inWords
                outWords = self.outWords
            
            newControl, self.contControl = self.control(controlInput, inWords, outWords, 
                self.questionLengths, control = control, contControl = self.contControl, name = cellName, reuse = cellReuse)
            
            # read unit
            # ablation: use whole question as control
            if config.controlWholeQ:                    
                newControl = self.vecQuestions

            info = self.read(self.knowledgeBase, memory, newControl, name = cellName, reuse = cellReuse) 

            if config.writeDropout < 1.0:
                # write unit
                info = tf.nn.dropout(info, self.dropouts["write"])
            
            newMemory = self.write(memory, info, newControl, self.contControl, name = cellName, reuse = cellReuse)
        
            # append as standard list?
            self.controls = tf.concat([self.controls, tf.expand_dims(newControl, axis = 1)], axis = 1)
            self.memories = tf.concat([self.memories, tf.expand_dims(newMemory, axis = 1)], axis = 1)
            self.infos = tf.concat([self.infos, tf.expand_dims(info, axis = 1)], axis = 1)

        newState = MACCellTuple(newControl, newMemory)
        return self.none, newState

    '''
    Initializes the a hidden state to based on the value of the initType:
    "PRM" for parametric initialization
    "ZERO" for zero initialization  
    "Q" to initialize to question vectors.

    Args:
        name: the state variable name.
        dim: the dimension of the state.
        initType: the type of the initialization
        batchSize: the batch size

    Returns the initialized hidden state.
    '''
    def initState(self, name, dim, initType, batchSize):
        if initType == "PRM":
            prm = tf.get_variable(name, shape = (dim, ),
                    initializer = tf.random_normal_initializer())                
            initState = tf.tile(tf.expand_dims(prm, axis = 0), [batchSize, 1])
        elif initType == "ZERO":
            initState = tf.zeros((batchSize, dim), dtype = tf.float32)
        else: # "Q"
            initState = self.vecQuestions
        return initState

    '''
    Add a parametric null word to the questions.

    Args:
        words: the words to add a null word to.
        [batchSize, questionLentgth]

        lengths: question lengths.
        [batchSize] 

    Returns the updated word sequence and lengths.  
    '''
    def addNullWord(self, words, lengths):
        nullWord = tf.get_variable("zeroWord", shape = (1 , config.ctrlDim), initializer = tf.random_normal_initializer())                    
        nullWord = tf.tile(tf.expand_dims(nullWord, axis = 0), [self.batchSize, 1, 1])
        words = tf.concat([nullWord, words], axis = 1)
        lengths += 1
        return words, lengths

    '''
    Initializes the cell internal state (currently it's stateful). In particular,
    1. Data-structures (lists of attention maps and accumulated losses).
    2. The memory and control states.
    3. The knowledge base (optionally merging it with the question vectors)
    4. The question words used by the cell (either the original word embeddings, or the 
       encoder outputs, with optional projection).

    Args:
        batchSize: the batch size

    Returns the initial cell state.
    '''
    def zero_state(self, batchSize, dtype = tf.float32):
        ## initialize data-structures
        self.attentions = {"kb": [], "question": [], "self": [], "gate": []}
        self.autoEncLosses = {"control": tf.constant(0.0), "memory": tf.constant(0.0)}


        ## initialize state
        initialControl = self.initState("initCtrl", config.ctrlDim, config.initCtrl, batchSize)
        initialMemory = self.initState("initMem", config.writeDim, config.initMem, batchSize)

        self.controls = tf.expand_dims(initialControl, axis = 1)
        self.memories = tf.expand_dims(initialMemory, axis = 1)
        self.infos = tf.expand_dims(initialMemory, axis = 1)
        
        self.contControl = initialControl

        ## initialize knowledge base
        # optionally merge question into knowledge base representation
        if config.initKBwithQ != "NON":
            if config.imageObjects:
                self.knowledgeBase = ops.linear(self.knowledgeBase, config.imageDims[-1], config.writeDim, name = "initKB")
                #print('inside zero state..')
                #print(self.knowledgeBase.get_shape())
            else:
                iVecQuestions = ops.linear(self.vecQuestions, config.ctrlDim, config.writeDim, name = "questions")

                concatMul = (config.initKBwithQ == "MUL")
                _, h, w, c, d = self.knowledgeBase.get_shape().as_list()
                knowledgeBase = tf.reshape(self.knowledgeBase, [-1, h*w, c*d])
                cnct, dim = ops.concat(knowledgeBase, iVecQuestions, config.writeDim, mul = concatMul, expandY = True)
                #print(cnct.get_shape())
                knowledgeBase = ops.linear(cnct, dim, config.writeDim, name = "initKB")
                #print(knowledgeBase.get_shape())
                self.knowledgeBase = tf.reshape(knowledgeBase, [-1, h, w, c, d])
                
        ## initialize question words
        # choose question words to work with (original embeddings or encoder outputs)
        words = self.questionCntxWords if config.controlContextual else self.questionWords    

        # optionally add parametric "null" word in the to all questions
        if config.addNullWord:
            words, self.questionLengths = self.addNullWord(words, self.questionLengths)

        # project words
        if config.controlPreDropout < 1.0:
            words = tf.nn.dropout(words, self.dropouts["controlPre"])

        self.inWords = self.outWords = words
        if config.controlInWordsProj or config.controlOutWordsProj:
            pWords = ops.linear(words, config.wrdQEmbDim, config.ctrlDim, name = "wordsProj")
            self.inWords = pWords if config.controlInWordsProj else words
            self.outWords = pWords if config.controlOutWordsProj else words

        ## initialize memory variational dropout mask
        if config.memoryVariationalDropout:
            self.memDpMask = ops.generateVarDpMask((batchSize, config.writeDim), self.dropouts["memory"])

        return MACCellTuple(initialControl, initialMemory)     
