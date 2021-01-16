"""
This code trains the JODIE model for the given dataset. 
The task is: interaction prediction.

How to run: 
$ python jodie.py --network reddit --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
"""

from library_data import *
import library_models as lib
from library_models import *
from datetime import datetime
import logging


logger = logging.getLogger('JODIE')
logger.setLevel(logging.DEBUG)
logger.propagate = False
handler = logging.FileHandler(filename='./log/'+datetime.strftime(datetime.now(), '%Y-%m-%d@%H-%M-%S')+'.log')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Let's go")

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', default="train", help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--embedding_dim_static', default=128, type=int, help='Number of dimensions of the static embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to False.')
args = parser.parse_args()

args.datapath = "data/%s.csv" % args.network 
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
# if args.gpu == -1:
#     args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# LOAD DATA
logger.info('Start loading data')
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1  # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
# true_labels_ratio = len(y_true)/(1.0+sum(y_true))  # +1 in denominator in case there are no state change labels, which will throw an error.
logger.info('Network statistics: %d users' % num_users)
logger.info('Network statistics: %d items' % num_items)
logger.info('Network statistics: %d interactions' % num_interactions)
print(("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true))))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
"""
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
"""
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = 300000  # timespan / 500

# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
# model = torch.nn.DataParallel(model, device_ids=[0])
# model = model.module
# weight = torch.Tensor([1, true_labels_ratio]).cuda()
# crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
# the initial user and item embeddings are learned during training as well
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

initial_user_embedding_static = nn.Parameter(F.normalize(torch.rand(args.embedding_dim_static).cuda(), dim=0))
initial_item_embedding_static = nn.Parameter(F.normalize(torch.rand(args.embedding_dim_static).cuda(), dim=0))
model.initial_user_embedding_static = initial_user_embedding_static
model.initial_item_embedding_static = initial_item_embedding_static

user_embeddings = initial_user_embedding.repeat(num_users, 1)  # initialize all users to the same embedding
item_embeddings = initial_item_embedding.repeat(num_items, 1)  # initialize all items to the same embedding
# item_embedding_static = Variable(torch.eye(num_items).cuda())  # one-hot vectors for static embeddings
# user_embedding_static = Variable(torch.eye(num_users).cuda())  # one-hot vectors for static embeddings
user_embeddings_static = initial_user_embedding_static.repeat(num_users, 1)
item_embeddings_static = initial_item_embedding_static.repeat(num_items, 1)

# INITIALIZE MODEL
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# optimizer = torch.nn.DataParallel(optimizer, device_ids=[0, 1])

logger.info('Try to save initial model before training')
save_model(model=model,
           optimizer=optimizer,
           args=args,
           epoch=-1,
           user_embeddings=torch.cat([user_embeddings, user_embeddings_static], dim=1),
           item_embeddings=torch.cat([item_embeddings, item_embeddings_static], dim=1),
           train_end_idx=train_end_idx)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
logger.info('Start training with %d epochs totally' % args.epochs)
print(("*** Training the JODIE model for %d epochs ***" % args.epochs))
try:
    with trange(args.epochs) as progress_bar1:
        for ep in progress_bar1:
            progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

            # INITIALIZE EMBEDDING TRAJECTORY STORAGE
            # user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim))
            # item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim))

            optimizer.zero_grad()
            reinitialize_tbatches()
            total_loss, loss, total_interaction_count = 0, 0, 0

            tbatch_start_time = None
            tbatch_to_insert = -1
            tbatch_full = False

            batch_count = 0

            # TRAIN TILL THE END OF TRAINING INTERACTION IDX
            try:
                with trange(train_end_idx) as progress_bar2:
                    for j in progress_bar2:
                        progress_bar2.set_description('Processed %dth interactions' % j)

                        # READ INTERACTION J
                        userid = user_sequence_id[j]
                        itemid = item_sequence_id[j]
                        feature = feature_sequence[j]
                        user_timediff = user_timediffs_sequence[j]
                        item_timediff = item_timediffs_sequence[j]

                        # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                        lib.tbatchid_user[userid] = tbatch_to_insert
                        lib.tbatchid_item[itemid] = tbatch_to_insert

                        lib.current_tbatches_user[tbatch_to_insert].append(userid)
                        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                        lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

                        timestamp = j
                        if tbatch_start_time is None:
                            tbatch_start_time = timestamp

                        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                        if timestamp - tbatch_start_time >= tbatch_timespan or timestamp == train_end_idx:
                            tbatch_start_time = timestamp  # RESET START TIME FOR THE NEXT TBATCHES

                            # ITERATE OVER ALL T-BATCHES
                            try:
                                with trange(len(lib.current_tbatches_user)) as progress_bar3:
                                    for i in progress_bar3:
                                        progress_bar3.set_description('Processed %d of %d T-batches in epoch-%d' % (i, len(lib.current_tbatches_user), ep))

                                        total_interaction_count += len(lib.current_tbatches_interactionids[i])

                                        # LOAD THE CURRENT TBATCH
                                        # Recall "lib.current_tbatches_user[i]" has unique elements
                                        tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                                        # Recall "lib.current_tbatches_item[i]" has unique elements
                                        tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                                        tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                                        # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                                        feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).cuda())
                                        user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                                        item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                                        tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                                        item_embedding_previous = item_embeddings[tbatch_itemids_previous, :]

                                        # PROJECT USER EMBEDDING TO CURRENT TIME

                                        user_embedding_input = user_embeddings[tbatch_userids, :]

                                        user_projected_embedding = model.forward(user_embeddings=user_embedding_input,
                                                                                 item_embeddings=item_embedding_previous,
                                                                                 timediffs=user_timediffs_tensor,
                                                                                 features=feature_tensor,
                                                                                 select='project')

                                        user_item_embedding = torch.cat([user_projected_embedding,
                                                                         item_embedding_previous,
                                                                         item_embeddings_static[tbatch_itemids_previous, :],
                                                                         user_embeddings_static[tbatch_userids, :]],
                                                                        dim=1)

                                        # PREDICT NEXT ITEM EMBEDDING
                                        predicted_item_embedding = model.forward(user_embeddings=user_item_embedding,
                                                                                 select='predict')

                                        # CALCULATE PREDICTION LOSS
                                        item_embedding_input = item_embeddings[tbatch_itemids, :]

                                        loss += MSELoss(predicted_item_embedding,
                                                        torch.cat([item_embedding_input,
                                                                   item_embeddings_static[tbatch_itemids, :]], dim=1).detach())

                                        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION

                                        user_embedding_output = model.forward(user_embeddings=user_embedding_input,
                                                                              item_embeddings=item_embedding_input,
                                                                              timediffs=user_timediffs_tensor,
                                                                              features=feature_tensor,
                                                                              select='user_update')
                                        item_embedding_output = model.forward(user_embeddings=user_embedding_input,
                                                                              item_embeddings=item_embedding_input,
                                                                              timediffs=item_timediffs_tensor,
                                                                              features=feature_tensor,
                                                                              select='item_update')

                                        item_embeddings[tbatch_itemids, :] = item_embedding_output
                                        user_embeddings[tbatch_userids, :] = user_embedding_output

                                        # user_embeddings_timeseries[tbatch_interactionids, :] = user_embedding_output.cpu()
                                        # item_embeddings_timeseries[tbatch_interactionids, :] = item_embedding_output.cpu()
                                        # user_embedding_output.cuda()
                                        # item_embedding_output.cuda()

                                        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                                        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                                        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                                        # CALCULATE STATE CHANGE LOSS
                                        # if args.state_change:
                                        #    loss += calculate_state_prediction_loss(model,
                                        #                                            tbatch_interactionids,
                                        #                                            user_embeddings_timeseries,
                                        #                                            y_true,
                                        #                                            crossEntropyLoss)
                            except KeyboardInterrupt:
                                progress_bar3.close()
                                raise
                            progress_bar3.close()

                            batch_count += 1
                            if batch_count % 4 == 0:
                                logger.info('Processed {:.2f}% in epoch-{:0>2d}, loss = {:.6f}'.format(
                                    j/train_end_idx * 100, ep, loss.item()
                                ))

                            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                            total_loss += loss.item()
                            # loss.mean()
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            optimizer.zero_grad()

                            # RESET LOSS FOR NEXT T-BATCH
                            loss = 0
                            item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
                            user_embeddings.detach_()
                            # item_embeddings_timeseries.detach_()
                            # user_embeddings_timeseries.detach_()

                            # REINITIALIZE
                            reinitialize_tbatches()
                            tbatch_to_insert = -1
            except KeyboardInterrupt:
                progress_bar2.close()
                raise
            progress_bar2.close()

            # END OF ONE EPOCH
            logger.info('Finish epoch-%d training with total loss = %f' % (ep, total_loss))
            print(("\n\nTotal loss in this epoch-%d = %f" % (ep, total_loss)))
            item_embeddings_dystat = torch.cat([item_embeddings, item_embeddings_static], dim=1)
            user_embeddings_dystat = torch.cat([user_embeddings, user_embeddings_static], dim=1)
            # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
            save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx)

            user_embeddings = initial_user_embedding.repeat(num_users, 1)
            item_embeddings = initial_item_embedding.repeat(num_items, 1)
except KeyboardInterrupt:
    progress_bar1.close()
    raise
progress_bar1.close()
# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
logger.info('Training complete and saving final model now')
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model,
           optimizer,
           args,
           ep,
           user_embeddings_dystat,
           item_embeddings_dystat,
           train_end_idx)

