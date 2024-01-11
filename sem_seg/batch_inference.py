import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')

parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--coarse_loss', type=float, default=0, help='Whether to use coarse loss')#1
parser.add_argument('--mmd_loss', type=float, default=0, help='Whether to use mmd loss')#200
parser.add_argument('--pfs', action='store_true', help='Whether to use pfs')
parser.add_argument('--fuct', action='store_true', help='Whether to use fully concated features')


FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir


COARSE_FLAG = FLAGS.coarse_loss
MMD_FLAG = FLAGS.mmd_loss
PFS_FLAG = FLAGS.pfs
FUCT_FLAG = FLAGS.fuct

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)]

NUM_CLASSES = 6

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)




def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        batch = tf.Variable(0, trainable=False)
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                                 batch*BATCH_SIZE,
                                                 BN_DECAY_DECAY_STEP,
                                                 BN_DECAY_DECAY_RATE,
                                                 staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

        # simple model
        pred, end_points = get_model(pointclouds_pl, labels_pl, is_training_pl, bn_decay, COARSE_FLAG, MMD_FLAG, PFS_FLAG, FUCT_FLAG)

        #print("11111111111111111111111111111111")

        loss = get_loss(pred, labels_pl, end_points, COARSE_FLAG, MMD_FLAG)

        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    
    total_correct = 0
    total_seen = 0

    fout_out_filelist = open(FLAGS.output_filelist, 'w')
    for room_path in ROOM_PATH_LIST:
        out_data_label_filename = os.path.basename(room_path)[:-4] + '_pred.txt'
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        out_gt_label_filename = os.path.basename(room_path)[:-4] + '_gt.txt'
        out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
        print(room_path, out_data_label_filename)
        a, b = eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename)
        total_correct += a
        total_seen += b
        fout_out_filelist.write(out_data_label_filename+'\n')
    fout_out_filelist.close()
    log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    if FLAGS.visu:
        fout = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_pred.obj'), 'w')
        fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_gt.obj'), 'w')
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')
    
    current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT)
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)
    # Get room dimension..
    data_label = np.load(room_path)
    data = data_label[:,0:6]
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN   0:12
        else:
            pred_label = np.argmax(pred_val, 2) # BxN
        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx+b, :, :]
            l = current_label[start_idx+b,:]
            pts[:,6] *= max_room_x
            pts[:,7] *= max_room_y
            pts[:,8] *= max_room_z
            pts[:,3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(NUM_POINT):
                color = indoor3d_util.g_label2color[pred[i]]
                color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
                if FLAGS.visu:
                    fout.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color[0], color[1], color[2]))
                    fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_gt[0], color_gt[1], color_gt[2]))
                fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (pts[i,6], pts[i,7], pts[i,8], pts[i,3], pts[i,4], pts[i,5], pred_val[b,i,pred[i]], pred[i]))
                fout_gt_label.write('%d\n' % (l[i]))
        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    fout_data_label.close()
    fout_gt_label.close()
    if FLAGS.visu:
        fout.close()
        fout_gt.close()
    return total_correct, total_seen


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
