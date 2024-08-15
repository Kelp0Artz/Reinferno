from func.DataPrep import DataPrep as dp
from func.FrontEnd import FrontEnd
from func.NeuralNetwork import *
from keras.datasets import mnist


"""
dataZ = [[0.9980489932812322, 0.9327281604917272], [0.5962770562129371, 0.5910994489994883], [0.5853642250471413, 0.6126339145934525], [0.9347524069564018, 0.7248462824798463], [0.8918328535261575, 0.5071842020114772], [0.6414132537265071, 0.6847597393821733], [0.7202007214601968, 0.7773380671930454], [0.6042946940632196, 0.5059907944080911], [0.9637844848127333, 0.6833914469470857], [0.7206581983400172, 0.5052634342672171], [0.8537618686966718, 0.8374496490533081], [0.8776084238428882, 0.536121104065305], [0.6107467022969057, 0.7859788129388774], [0.7373059163818658, 0.6741272482748134], [0.6322367521263183, 0.8543586623498833], [0.8130818429994374, 0.5018309702137424], [0.764849610028002, 0.6275377211373337], [0.7104177439112141, 0.6753792624196758], [0.816820278467599, 0.7430839863030137], [0.5998400663674126, 0.6612553017386092], [0.8640873747459758, 0.5474122890420417], [0.9485079411738218, 0.5864571906822073], [0.7455665437215478, 0.6070648678463968], [0.6079489815424197, 0.8403383842432101], [0.9152651591369513, 0.5310769247160241], [0.947404869517838, 0.6808230939331771], [0.9367798785021437, 0.7101242441425542], [0.7326617326248744, 0.7062757879154671], [0.9293306434370043, 0.8303405270493472], [0.6078357800850056, 0.5771807213898521], [0.8117301699634303, 0.7180657698007557], [0.665242049220727, 0.5984163588124886], [0.9177170160580517, 0.6813879947522165], [0.8595881386217061, 0.8339982011580513], [0.7444164200380885, 0.6537757752640552], [0.9192206932181238, 0.6601881111707157], [0.8084015766691934, 0.6441506067272938], [0.6224719070244454, 0.7693152739869648], [0.8411531594107934, 0.6315735463296742], [0.7456718050559916, 0.6311456587171615], [0.7029970896688777, 0.6247016392308357], [0.8772816935304171, 0.5249648428037481], [0.599198870366078, 0.8200719991331543], [0.8544225641684866, 0.5831346040845921], [0.5988039945288134, 0.817214402773351], [0.5982083522928698, 0.8209136904578694], [0.8369400250272546, 0.6957768031918213], [0.7590088683140731, 0.632611912761301], [0.8741296937847058, 0.827593716511234], [0.8793172437237333, 0.7315666210164302], [0.9411132242674314, 0.6790542835101294], [0.908932010129385, 0.726091148726888], [0.7152184511543997, 0.6933580702127126], [0.6930550877997433, 0.674968215230996], [0.9296664184916867, 0.6848950215869423], [0.9058605228685291, 0.7641462745733387], [0.8472658976730246, 0.6017789538708292], [0.6160544519459608, 0.5952905587214212], [0.873064099122343, 0.6142002327412208], [0.8771930904027137, 0.7697526666162387], [0.7539515745645645, 0.5704296458106148], [0.7687482113047275, 0.630141615075937], [0.8491011741973963, 0.5862174372764864], [0.6143041578420045, 0.5313618305294523], [0.8222898727818688, 0.7070281193548276], [0.7436182023809057, 0.8342760121360494], [0.8766466420606598, 0.7864799417844417], [0.8914023150077834, 0.8147976403707208], [0.7320423612257836, 0.6846316988621641], [0.8941847251305597, 0.5294682816955032], [0.6688977376220297, 0.5893068601361942], [0.799587008857738, 0.5834402222742425], [0.6823137435256182, 0.7543265561157464], [0.7421826535679463, 0.8250452176624185], [0.8200734245894154, 0.5611185921229394], [0.7254822586416765, 0.6837460387427852], [0.830663869582362, 0.5165061595924193], [0.8808978085676082, 0.6225484417715427], [0.7210151489188543, 0.7594984742583774], [0.9113007463062138, 0.5461590200721712], [0.7818441668403134, 0.8212657044534594], [0.8200348235125653, 0.7525145053539866], [0.7576993256174855, 0.564941002231468], [0.8877644565993516, 0.6039521405985092], [0.7374977081499361, 0.7601977839511238], [0.605312350284993, 0.6582641537468882], [0.8029073902449954, 0.6921194137840022], [0.8516439122210571, 0.5609407897832821], [0.9441420460836094, 0.6866551589997695], [0.8767032742717992, 0.6545938714858977], [0.776313423388142, 0.571470269065795], [0.9210550725634567, 0.5547344441187891], [0.6352667413832658, 0.7262426112420656], [0.7207280987807245, 0.7753126932878304], [0.7123935398970104, 0.6395941565991458]]
dataX = [[0.9980489932812322, 0.9327281604917272], [0.5962770562129371, 0.5910994489994883], [0.5853642250471413, 0.6126339145934525], [0.9347524069564018, 0.7248462824798463], [0.8918328535261575, 0.5071842020114772], [0.6414132537265071, 0.6847597393821733], [0.7202007214601968, 0.7773380671930454], [0.6042946940632196, 0.5059907944080911], [0.9637844848127333, 0.6833914469470857], [0.7206581983400172, 0.5052634342672171], [0.8537618686966718, 0.8374496490533081], [0.8776084238428882, 0.536121104065305], [0.6107467022969057, 0.7859788129388774], [0.7373059163818658, 0.6741272482748134], [0.6322367521263183, 0.8543586623498833], [0.8130818429994374, 0.5018309702137424], [0.764849610028002, 0.6275377211373337], [0.7104177439112141, 0.6753792624196758], [0.816820278467599, 0.7430839863030137], [0.5998400663674126, 0.6612553017386092], [0.8640873747459758, 0.5474122890420417], [0.9485079411738218, 0.5864571906822073], [0.7455665437215478, 0.6070648678463968], [0.6079489815424197, 0.8403383842432101], [0.9152651591369513, 0.5310769247160241], [0.947404869517838, 0.6808230939331771], [0.9367798785021437, 0.7101242441425542], [0.7326617326248744, 0.7062757879154671], [0.9293306434370043, 0.8303405270493472], [0.6078357800850056, 0.5771807213898521], [0.8117301699634303, 0.7180657698007557], [0.665242049220727, 0.5984163588124886], [0.9177170160580517, 0.6813879947522165], [0.8595881386217061, 0.8339982011580513], [0.7444164200380885, 0.6537757752640552], [0.9192206932181238, 0.6601881111707157], [0.8084015766691934, 0.6441506067272938], [0.6224719070244454, 0.7693152739869648], [0.8411531594107934, 0.6315735463296742], [0.7456718050559916, 0.6311456587171615], [0.7029970896688777, 0.6247016392308357], [0.8772816935304171, 0.5249648428037481], [0.599198870366078, 0.8200719991331543], [0.8544225641684866, 0.5831346040845921], [0.5988039945288134, 0.817214402773351], [0.5982083522928698, 0.8209136904578694], [0.8369400250272546, 0.6957768031918213], [0.7590088683140731, 0.632611912761301], [0.8741296937847058, 0.827593716511234], [0.8793172437237333, 0.7315666210164302], [0.9411132242674314, 0.6790542835101294], [0.908932010129385, 0.726091148726888], [0.7152184511543997, 0.6933580702127126], [0.6930550877997433, 0.674968215230996], [0.9296664184916867, 0.6848950215869423], [0.9058605228685291, 0.7641462745733387], [0.8472658976730246, 0.6017789538708292], [0.6160544519459608, 0.5952905587214212], [0.873064099122343, 0.6142002327412208], [0.8771930904027137, 0.7697526666162387], [0.7539515745645645, 0.5704296458106148], [0.7687482113047275, 0.630141615075937], [0.8491011741973963, 0.5862174372764864], [0.6143041578420045, 0.5313618305294523], [0.8222898727818688, 0.7070281193548276], [0.7436182023809057, 0.8342760121360494], [0.8766466420606598, 0.7864799417844417], [0.8914023150077834, 0.8147976403707208], [0.7320423612257836, 0.6846316988621641], [0.8941847251305597, 0.5294682816955032], [0.6688977376220297, 0.5893068601361942], [0.799587008857738, 0.5834402222742425], [0.6823137435256182, 0.7543265561157464], [0.7421826535679463, 0.8250452176624185], [0.8200734245894154, 0.5611185921229394], [0.7254822586416765, 0.6837460387427852], [0.830663869582362, 0.5165061595924193], [0.8808978085676082, 0.6225484417715427], [0.7210151489188543, 0.7594984742583774], [0.9113007463062138, 0.5461590200721712], [0.7818441668403134, 0.8212657044534594], [0.8200348235125653, 0.7525145053539866], [0.7576993256174855, 0.564941002231468], [0.8877644565993516, 0.6039521405985092], [0.7374977081499361, 0.7601977839511238], [0.605312350284993, 0.6582641537468882], [0.8029073902449954, 0.6921194137840022], [0.8516439122210571, 0.5609407897832821], [0.9441420460836094, 0.6866551589997695], [0.8767032742717992, 0.6545938714858977], [0.776313423388142, 0.571470269065795], [0.9210550725634567, 0.5547344441187891], [0.6352667413832658, 0.7262426112420656], [0.7207280987807245, 0.7753126932878304], [0.7123935398970104, 0.6395941565991458]]
dataX = np.array(dataX)
print(dataX.shape)
slopes =[0.42307273, 0.47859815]
dataY = []

#dataZ = [[1,1],[1,1]]
#dataX = [[0,0],[0,0]]

for a in dataX:
    for i in range(len(a)):
        a[i] = a[i] * slopes[i]
    dataY.append(a)
print(dataY)  
model = LinearRegression()
model.fit(X_train, Y_train)
model.evaluate(X_test, Y_test)
model.predict(X_test[0])
dataX = np.array(dataX)

model = LinearRegression()
model.fit(dataX, dataY)


def leastSquares(dataX, dataY, learningRate):
    listOfSquaredResiduals = np.zeros()
    squaredResidual = 0
    slope = 1
    intercept = 0
    NUM_ITERATION = 1000
    iteration = 0
    def calculateSquaredResidual(dataX, dataY, slope, intercept):
        residual = 0
        for i in range(len(dataX)):
            squaredResidual += ( dataY - (slope * dataX[i] + intercept))**2
        return squaredResidual
    
    while squaredResidual > 0.0001 or NUM_ITERATION > iteration:
        intercept = intercept - 0.1
        squaredResidual = calculateSquaredResidual(dataX, dataY, slope, intercept)
        iteration += 1
        slope -= slope*learningRate
    np.append(listOfSquaredResiduals,calculateSquaredResidual(dataX, dataY, slope, intercept))"""


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)




X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# X_train[num_row, num_column]
model = NeuralNetwork()

model.add(Layer.create_layer(model, 2, input_shape = 10)) 
model.add(Layer.create_layer(model, 8))
model.add(Layer.create_layer(model, 1))
print(model.layers[1])
#print(np.shape(model.layers[0][0][0]))
#model.settings()
#model.fit(X_train, Y_train, num_batches= 10)
#print(model.activation_layers)
#print(model.vector_multiply(X_train, 0, 0))
#model.forward_propagation(X_train[0])