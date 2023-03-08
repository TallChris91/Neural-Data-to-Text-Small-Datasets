import os

def train_command(currentpath, language, category, addition, testpart):
    if addition == 'None':
        if language == 'en':
            os.system('python3 train.py --tokenizer t5-large --model t5-large --src_train CACAPO/Input/' + language + '/' + category + '/trainsrc.txt --trg_train CACAPO/Input/' + language + '/' + category + '/traintrg.txt --src_dev CACAPO/Input/' + language + '/' + category + '/devsrc.txt --trg_dev CACAPO/Input/' + language + '/' + category + '/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/base --language english --verbose --batch_status 16 --cuda')
        elif language == 'nl':
            os.system(
                'python3 train.py --tokenizer google/mt5-large --model google/mt5-large --src_train CACAPO/Input/' + language + '/' + category + '/trainsrc.txt --trg_train CACAPO/Input/' + language + '/' + category + '/traintrg.txt --src_dev CACAPO/Input/' + language + '/' + category + '/devsrc.txt --trg_dev CACAPO/Input/' + language + '/' + category + '/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/base --language nl --verbose --batch_status 16 --cuda')
    else:
        if language == 'en':
            os.system('python3 train.py --tokenizer t5-large --model t5-large --src_train CACAPO/Input/' + language + '/' + category + '/' + addition + '_train' + testpart + 'src.txt --trg_train CACAPO/Input/' + language + '/' + category + '/' + addition + '_train' + testpart + 'trg.txt --src_dev CACAPO/Input/' + language + '/' + category + '/devsrc.txt --trg_dev CACAPO/Input/' + language + '/' + category + '/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + ' --language english --verbose --batch_status 16 --cuda')
        elif language == 'nl':
            os.system(
                'python3 train.py --tokenizer google/mt5-large --model google/mt5-large --src_train CACAPO/Input/' + language + '/' + category + '/' + addition + '_train' + testpart + 'src.txt --trg_train CACAPO/Input/' + language + '/' + category + '/' + addition + '_train' + testpart + 'trg.txt --src_dev CACAPO/Input/' + language + '/' + category + '/devsrc.txt --trg_dev CACAPO/Input/' + language + '/' + category + '/devtrg.txt --epochs 16 --learning_rate 1e-5 --batch_size 2 --early_stop 5 --max_length 180 --write_path ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + ' --language nl --verbose --batch_status 16 --cuda')

def evaluate_command(currentpath, language, category, devtest, addition, testpart):
    if addition == 'None':
        if language == 'en':
            os.system('python3 evaluate.py --tokenizer t5-large --model ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/base/model --src_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'src.txt --trg_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/CACAPO/' + language + '/' + category + '/base/' + devtest + ' --language english --verbose --batch_status 16 --cuda')
        elif language == 'nl':
            os.system('python3 evaluate.py --tokenizer google/mt5-large --model ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/base/model --src_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'src.txt --trg_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/CACAPO/' + language + '/' + category + '/base/' + devtest + ' --language nl --verbose --batch_status 16 --cuda')
    else:
        if language == 'en':
            os.system('python3 evaluate.py --tokenizer t5-large --model ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + '/model --src_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'src.txt --trg_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + '/' + devtest + ' --language english --verbose --batch_status 16 --cuda')
        elif language == 'nl':
            os.system(
                'python3 evaluate.py --tokenizer google/mt5-large --model ' + currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + '/model --src_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'src.txt --trg_test CACAPO/Input/' + language + '/' + category + '/' + devtest + 'trg.txt --batch_size 8 --max_length 180 --write_dir ' + currentpath + '/results/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + '/' + devtest + ' --language nl --verbose --batch_status 16 --cuda')

currentpath = os.getcwd()

#languagelist = ['en', 'nl']
languagelist = ['en']
for language in languagelist:
    categorylist = ['Accidents', 'Sports', 'Stocks', 'Weather']
    for category in categorylist:
        additionlist = ['None', 'dat_aug', 'sem_par']
        for addition in additionlist:
            if addition == 'None':
                os.makedirs(currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/base', exist_ok=True)
                train_command(currentpath, language, category, addition, '')
                devtestlist = ['dev', 'test']
                for devtest in devtestlist:
                    os.makedirs(currentpath + '/results/CACAPO/' + language + '/' + category + '/base/' + devtest, exist_ok=True)
                    evaluate_command(currentpath, language, category, devtest, addition, '')
            else:
                testpartlist = ['125', '250', '500', '1000']
                for testpart in testpartlist:
                    os.makedirs(currentpath + '/SavedModels/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart, exist_ok=True)
                    train_command(currentpath, language, category, addition, testpart)
                    devtestlist = ['dev', 'test']
                    for devtest in devtestlist:
                        os.makedirs(currentpath + '/results/CACAPO/' + language + '/' + category + '/' + addition + '/' + testpart + '/' + devtest, exist_ok=True)
                        evaluate_command(currentpath, language, category, devtest, addition, testpart)
