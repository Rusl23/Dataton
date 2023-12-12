from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# model block

import numpy as np
import pandas as pd
import pickle

TOKEN: Final = '6730826958:AAEnVW_2n4AxkNHKp-G4t610SQV1tvnf6Zc'
BOT_USERNAME: Final = '@Solar_rad_bot'


# commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Привет! Я бот, предсказываю солнечную радиацию')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
      """
      /start -> Привет! Я бот, предсказываю солнечную радиацию!
      /help -> Напечатайте Привет, чтобы начать предсказание
      /Start_prediction -> Введите Температуру, Давление, Валажность, Направление ветра, Скорость ветра, Время: 59, 30.4, 84, 68.92, 5.62, 13
      """)

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Введите: \n \
    1.Температуру \n \
    2. Атмосферное давление \n \
    3. Влажность \n \
    4. Направление ветра(в градусах) \n \
    5. Скорость ветра (м/с) \n \
    6. Время суток (в часах, например 14) \n \
    7. Разделитель точка \n \
* Пожалуйста вводите только цифры, как в примере:  59, 30.4, 84, 68.92, 5.62, 13')

# responses
def handle_response(text: str)->str:
    processed: str = text.lower()

    if 'привет' in processed:
        return 'Введите: \n \
    1.Температуру (в цельсиях) \n \
    2. Атмосферное давление (мм рт.ст) \n \
    3. Влажность (%) \n \
    4. Направление ветра (восточный, юго-восточный... или в градусах) \n \
    5. Скорость ветра (м/с) \n \
    6. Время суток (в часах, например 14) \n \
    7. Разделитель точка \n \
* Пожалуйста вводите только цифры (кроме направлнеия ветра), как в примере:  59, 30.4, 84, 68.92, 5.62, 13'
    s = get_predict(text)
    return s

def get_predict(text):
    
    chat_data = text.split(', ')
    if len(chat_data) != 6:
        return 'Ошибка ввода: нужно ввести 6 параметров'
    try:
        if chat_data[3] == 'северный':
            chat_data[3] = 360
        elif chat_data[3] == 'южный':
            chat_data[3] = 180
        elif chat_data[3] == 'восточный':
            chat_data[3] = 90
        elif chat_data[3] == 'западный':
            chat_data[3] = 270
        elif chat_data[3] == 'северо-восточный':
            chat_data[3] = 45
        elif chat_data[3] == 'юго-восточный':
            chat_data[3] = 225
        elif chat_data[3] == 'северо-западный':
            chat_data[3] = 315
        elif chat_data[3] == 'юго-западный':
            chat_data[3] = 225
        else:
            chat_data[3] = chat_data[3]
        mod_data = [float(num) for num in chat_data]
        mod_data[0] = (mod_data[0] * 9/5) + 32
        mod_data[1] = mod_data[1] * 0.03937
        predict_sol = get_predict_solrad(mod_data)
        return predict_sol
    except ValueError:
        return 'Ошибка ввода: пожалуйста введите данные, как указано в примере'

# model block
def get_predict_solrad(mod_data): # predict solar ratiation watts per meter^2 by model CatBoostRegressor
    data_arg = pd.DataFrame([mod_data],
                            columns=['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'hour'])
    with open('cbr_scaler.pkl', 'rb') as file_scl:
        scaler = pickle.load(file_scl)
    data_arg = scaler.transform(data_arg)
    return predict_solrad_cbr(data_arg)

def predict_solrad_cbr(data_arg):
    with open('cbr_model.pkl', 'rb') as file_model:
        model =  pickle.load(file_model)
    predict = model.predict(data_arg)
    predict_mod = np.exp(predict)
    return interprit(predict_mod)

def interprit(predict_mod):
    s_mod = round(float(predict_mod),2)
    if s_mod < 100:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Низкая интенсивность. Минимальное воздействие на кожу. Обычно соответствует тусклому свету или рассеянному солнечному свету.'
    elif 100 <= s_mod < 200:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Умеренная интенсивность. Воздействие на кожу в пределах нормы.'
    elif 200 <= s_mod < 400:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Средняя интенсивность. Умеренное воздействие на кожу. Осторожность рекомендуется, особенно для людей с чувствительной кожей.'
    elif 400 <= s_mod < 600:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Повышенная интенсивность. Значительное воздействие на кожу. Рекомендуется использование солнцезащитных средств и предосторожность'
    elif 600 <= s_mod < 800:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Высокая интенсивность. Воздействие на кожу значительно выше нормы. Необходимость принятия дополнительных мер предосторожности'
    elif 800 <= s_mod < 1000:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Очень высокая интенсивность. Высокий риск солнечных ожогов и других негативных эффектов. Использование солнцезащитных средств обязательно'
    else:
        return f'модель предсказывает солнечную радиацию {s_mod} ватт на м2. Экстремально высокая интенсивность. Опасность сильного воздействия на кожу. Рекомендуется избегать пребывания на солнце в этот период'



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)

    print('Bot:', response)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('Starting bot ...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    # Pols the bot
    print('polling ...')
    app.run_polling(poll_interval=3)



