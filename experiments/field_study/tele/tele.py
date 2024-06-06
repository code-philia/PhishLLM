from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import PollHandler
from gsheets import gwrapper
import logging
import os
from telegram.error import RetryAfter
from telegram.ext.dispatcher import run_async
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
import time
REQUEST_KWARGS = {"proxy_url": "http://127.0.0.1:7890"}

class teleBot():
    def __init__(self):
        updater = Updater(token=open('/home/ruofan/git_space/ScamDet/datasets/telegram_token.txt').read(),
                          use_context=True,
                          request_kwargs=REQUEST_KWARGS)
        self.folder_to_label = "/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2024-01-29"
        # self.folder_to_label = "/home/ruofan/git_space/ScamDet/datasets/field_study/2023-08-11"

        self.dict_date = {}
        dispatcher = updater.dispatcher
        self.sheets = gwrapper()
        print(len(os.listdir(self.folder_to_label)))

        start_handler = CommandHandler('start', self.start)
        dispatcher.add_handler(start_handler)

        poll_handler = PollHandler(self.poll)
        dispatcher.add_handler(poll_handler)

        stat_handler = CommandHandler('get', self.get_stats)
        dispatcher.add_handler(stat_handler)
        updater.start_polling()
        updater.idle()

    def poll(self, update, context):

        print(update)
        question_id = update['poll']['question'].split('~')[0]
        yes = update['poll']['options'][0]['voter_count']
        no = update['poll']['options'][1]['voter_count']
        unsure = update['poll']['options'][2]['voter_count']
        try:
            self.sheets.update_cell(question_id, yes, no, unsure)
            print('update happens')
        except Exception as e:
            print(e)
            time.sleep(5)
            self.poll(update, context)

    def get_stats(self, update, context):

        rows = self.sheets.get_records()
        date = update.message.text.split(' ')[1]
        print(date)
        print(len(rows))
        unanswered = 0
        ambigious = 0
        phishing = 0
        non_phishing = 0
        unsure = 0
        for i in range(len(rows)):
            row = rows[i]

            if date == 'all' or date == row['date']:
                if row['unsure'] != 0:
                    unsure += 1
                elif row['yes'] > 0 and row['no'] == 0:
                    phishing += 1
                elif row['yes'] == 0 and row['no'] > 0:
                    non_phishing += 1
                elif row['yes'] == 0 and row['no'] == 0:
                    unanswered += 1
                elif row['yes'] > 0 and row['no'] > 0:
                    ambigious += 1

        message = "unaswered:{}\n ambigious:{}\n phishing:{}\n nonphishing:{} \n unsure:{}".format(str(unanswered),
                                                                                                   str(ambigious),
                                                                                                   str(phishing),
                                                                                                   str(non_phishing),
                                                                                                   str(unsure))
        context.bot.send_message(chat_id=update.effective_chat.id, text=message)

    def update_file(self):
        rows = self.sheets.get_records()
        folder_names = list(map(lambda x: x['foldername'], rows))
        base = self.folder_to_label
        to_update = []

        # for i in os.listdir(base):
        #     folder = os.path.join(base, i)
        #     print(folder)
        #     for j in os.listdir(folder):
        #         data_folder = os.path.join(folder, j)
        #         if j in folder_names:
        #             continue
        #         else:
        #             print(j)
        #             info_file = os.path.join(data_folder, 'info.txt')
        #             if os.path.exists(info_file):
        #                 with open(info_file, 'r') as f:
        #                     url = f.readline()
        #             else:
        #                 url = "Cannot find info file"
        #             to_update.append([i, url, j, 0, 0, 0, 0, 0, 0])

        # label one-day data only
        for i in os.listdir(base):
            date = os.path.basename(base)
            data_folder = os.path.join(base, i)
            if i in folder_names:
                continue
            else:
                print(i)
                info_file = os.path.join(data_folder, 'info.txt')
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        url = f.readline()
                else:
                    url = "Cannot find info file"
                to_update.append([date, url, i, 0, 0, 0, 0, 0, 0])

        self.sheets.update_list(to_update)

    def start(self, update, context):

        context.bot.send_message(chat_id=update.effective_chat.id, text="Getting files from sheets")
        self.update_file()
        base = self.folder_to_label

        context.bot.send_message(chat_id=update.effective_chat.id, text="Time to get to work ")
        rows = self.sheets.get_records()
        for i in range(len(rows)):
            row = rows[i]
            if row['yes'] == 0 and row['no'] == 0 and row['unsure'] == 0:
                # folder_path = os.path.join(base, row['date'], row['foldername'])
                folder_path = os.path.join(base, row['foldername'])
                path = os.path.join(folder_path, 'predict.png')
                if not os.path.exists(path):
                    path = os.path.join(folder_path, 'shot.png')

                try:
                    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(path, 'rb'))
                    context.bot.send_poll(chat_id=update.effective_chat.id, options=['yes', 'no', 'unsure'],
                                          question=str(i + 2) + '~' + row['url'][:100], )
                    time.sleep(5)
                except RetryAfter as e:
                    time.sleep(60)
                    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(path, 'rb'))
                    context.bot.send_poll(chat_id=update.effective_chat.id, options=['yes', 'no', 'unsure'],
                                          question=str(i + 2) + '~' + row['url'][:100], )
                except Exception as e:
                    print(e)
                    context.bot.send_message(chat_id=update.effective_chat.id, text='unable to display image')
                    context.bot.send_poll(chat_id=update.effective_chat.id, options=['yes', 'no', 'unsure'],
                                          question=str(i + 2) + '~' + row['url'][:100], )

    def _map_date_to_folder(self, date):
        return str(date)


if __name__ == '__main__':
    teleBot()

