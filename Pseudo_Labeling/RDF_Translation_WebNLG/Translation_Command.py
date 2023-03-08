import os

def train_command(currentpath, category):
    os.system('python3 run_translation.py --model_name_or_path t5-large --do_train --do_eval --do_predict --source_lang en --target_lang data --source_prefix "webnlg: " --train_file ' + currentpath + '/Data/' + category + '/' + category + '_train.json --validation_file ' + currentpath + '/Data/' + category + '/' + category + '_dev.json --test_file ' + currentpath + '/Data/' + category + '/' + category + '_all_predictions.json --output_dir ' + currentpath + '/Model/' + category + '-translation --num_train_epochs=30 --per_device_train_batch_size=2 --per_device_eval_batch_size=8 --overwrite_output_dir --predict_with_generate')


currentpath = os.getcwd()
subdirlist = os.listdir(currentpath + '/Data')

for category in subdirlist:
    train_command(currentpath, category)