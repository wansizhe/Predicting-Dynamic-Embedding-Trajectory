from library_data import *
from library_models import *
from datetime import datetime
import logging


logger = logging.getLogger('JODIE')
logger.setLevel(logging.DEBUG)
logger.propagate = False
handler = logging.FileHandler(filename='./log/Predict-'+datetime.strftime(datetime.now(), '%Y-%m-%d@%H-%M-%S')+'.log')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Let's go")


parser = argparse.ArgumentParser()
parser.add_argument('--network', default='test', help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--epoch', default=7, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--embedding_dim_static', default=128, type=int, help='Number of dimensions of the static embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load user id to predict
logger.info('Start loading data')
[user2id,
 user_sequence_id,
 user_timediffs_sequence,
 user_previous_itemid_sequence,
 item2id,
 item_sequence_id,
 item_timediffs_sequence,
 timestamp_sequence,
 feature_sequence,
 y_true,
 user_last,
 item_last] = load_network_test(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1

id2user = dict(zip(user2id.values(), user2id.keys()))

logger.info('Network statistics: %d users' % num_users)
logger.info('Network statistics: %d items' % num_items)

logger.info('Loading model')

# load trained model
model = JODIE(args, num_features, num_users, num_items).cuda()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)

logger.info('Loading embeddings')

item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
item_embeddings = item_embeddings.clone()
item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
item_embeddings_static = item_embeddings_static.clone()

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
user_embeddings_static = user_embeddings_static.clone()

# user_pred = np.load('../../ydzhang/CIKM2019/data/jodie_input_all_9.npy')
logger.info('Loading users')
user_pred = []
with open('../../ydzhang/CIKM2019/data/ECommAI_ubp_round1_test', 'r') as f:
    for i, l in enumerate(f):
        user_id = int(l)
        user_pred.append(user_id)

user_pred = np.array(user_pred)

logger.info('Loading popular items')
pop_items = np.load(os.path.join('../../ydzhang/CIKM2019/data', "hot_item_5k.npy"))
pop_items = pop_items.reshape(-1, )

prediction = []
logger.info('Start predicting')
try:
    with trange(len(user_pred)) as progress_bar:
        for j in progress_bar:
            progress_bar.set_description('%dth user to predict' % j)

            # Get user
            if user_pred[j] not in user2id.keys():
                curr_prediction = np.array(popular_recommend(pop_items, 50))
            else:
                userid = user2id[user_pred[j]]

                user_timediff = user_timediffs_sequence[user_last[user_pred[j]]]  # user_timediffs_sequence[j]
                # item_timediff = item_timediffs_sequence[item_last]# item_timediffs_sequence[j]

                itemid_previous = item_sequence_id[user_last[user_pred[j]]]
                user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
                user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]

                feature_tensor = Variable(torch.Tensor([1, 0, 0, 0]).cuda()).unsqueeze(0)
                user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
                # item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
                item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

                item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]
                user_projected_embedding = model.forward(user_embedding_input,
                                                         item_embedding_previous,
                                                         timediffs=user_timediffs_tensor,
                                                         features=feature_tensor,
                                                         select='project')
                user_item_embedding = torch.cat([user_projected_embedding,
                                                 item_embedding_previous,
                                                 item_embeddings_static[torch.cuda.LongTensor([itemid_previous])],
                                                 user_embedding_static_input], dim=1)

                # PREDICT ITEM EMBEDDING
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1),
                                                            torch.cat([item_embeddings, item_embeddings_static],
                                                                      dim=1)).squeeze(-1)
                euclidean_distances, rank = euclidean_distances.sort(descending=False)
                idx = torch.arange(num_items).cuda()
                # top50_id = idx[torch.lt(euclidean_distances, euclidean_distances[rank==50])]
                top50_id = idx[rank < 50].data.cpu().numpy()

                curr_prediction = [id2user[top50_id[i]] for i in range(50)]
                # curr_prediction.append(user_pred[j])

            prediction.append(curr_prediction)
            logger.info('%dth user=%d, prediction=[%d, %d, %d], [%d], shape=%dx%d' % (j, user_pred[j], curr_prediction[0], curr_prediction[1], curr_prediction[2], curr_prediction[-1], len(prediction), len(curr_prediction)))
except KeyboardInterrupt:
    progress_bar.close()
    raise
progress_bar.close()

np.save('prediction_result_3.npy', np.array(prediction))
