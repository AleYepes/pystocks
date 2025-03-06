import pandas as pd
from time import sleep
from datetime import datetime, timedelta
import logging
import traceback
import ast
import random
from tqdm import tqdm
import pyautogui
import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz import fuzz
import gc
import re
from PIL import Image
import cv2

import platform
if platform.system() == "Darwin":
    from AppKit import NSWorkspace, NSApplicationActivateIgnoringOtherApps


# Set-up functions
def extract_text(ocr, screenshot):
    try:
        result = ocr.ocr(screenshot, cls=False)
    except Exception as e:
        print(f"Error during OCR: {e}")
        result = [None]

    text = []
    if result[0]:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                if line[-1][-1] > 0.85:
                    text.append(line[-1][0].replace(' ',''))
    return text

def scroll(amount, attempts=2):
    SCROLL_TOP = 600
    SCROLL_BOT = 550
    
    before = pyautogui.screenshot(region=(0, 25, 1919, 1054))
    before = np.array(before)
    for _ in range(attempts):
        rand_int = random.randint(SCROLL_BOT, SCROLL_TOP)
        pyautogui.scroll(amount, rand_int, rand_int)
        after = pyautogui.screenshot(region=(0, 25, 1919, 1054))
        after = np.array(after)

        if not np.array_equal(before, after):
            return

def wait(seconds=5, interval=0.5):
    start = datetime.now()
    timeout = timedelta(seconds=seconds)
    sleep(interval)
    while datetime.now() - start < timeout:
        screenshot = pyautogui.screenshot(region=(10, 280, 440, 200))
        screenshot = np.array(screenshot)
        screenshot = screenshot[:, :, :3]

        if np.any(np.all(screenshot == (247, 247, 247), axis=-1)):
            return
        sleep(interval)
    raise Exception('wait() text never loaded')

def select_tab(name, seconds=5):
    try:
        scroll(999)
        tab = pyautogui.locateCenterOnScreen(f'assets/{name}/tab_label.png', confidence=0.9)
        pyautogui.click(tab)
        wait(seconds)
        return True
    except Exception:
        return False

def clear_modals():
    while True:
        try:
            log_off_timer = pyautogui.locateOnScreen("assets/fund_setup/log_off_timer.png", confidence=0.9)
            left_x = log_off_timer.left + 34
            center_y = log_off_timer.top + log_off_timer.height // 2
            pyautogui.click(left_x, center_y)
        except Exception:
            break

    while True:
        try:
            agree = pyautogui.locateOnScreen("assets/fund_setup/agree.png", confidence=0.9)
            pyautogui.click(agree)
        except Exception:
            break

    while True:
        try:
            search_error = pyautogui.locateOnScreen("assets/fund_setup/search_error.png", confidence=0.9)
            left_x = search_error.left + 25
            center_y = search_error.top + search_error.height // 2
            pyautogui.click(left_x, center_y)
        except Exception:
            break

def select_exchange():
    while True:
        try:
            pyautogui.locateOnScreen("assets/fund_setup/contract_selection.png", confidence=0.9)
            pyautogui.press("up")
            pyautogui.press("enter")
            return True
        except Exception:
            break

def switch_to_app(app_name='Trader Workstation'):
    workspace = NSWorkspace.sharedWorkspace()
    apps = workspace.runningApplications()
    for app in apps:
        if app.localizedName() == app_name:
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            break

def check_search_results(ocr, row, screenshot, screenshot_left, screenshot_top, buffer, width=790):
    matches, HEIGHT, adjustable_height, max_adjustable_height, top, text_detected = [], 21, 21, 42, buffer, False

    screenshot_array = np.array(screenshot)
    target_color = [64, 64, 64, 255]    
    replacement_color = [109, 111, 113, 255]
    mask = np.all(screenshot_array == target_color, axis=-1)
    screenshot_array[mask] = replacement_color
    
    while True:
        text_list = extract_text(ocr, screenshot_array[top:top+adjustable_height])

        if not text_list:
            if text_detected:
                pass
            else:
                break

        if len(text_list) > 1:
            search_symbol = text_list[0]
            row_symbol = str(row['symbol'])
            if (len(search_symbol) == len(row_symbol)) and fuzz.partial_ratio(search_symbol, row_symbol) >= 75:
                search_exchange = text_list[-1]
                position = (screenshot_left + (width / 2), (screenshot_top + top) + adjustable_height / 2)
                matches.append((search_exchange, search_symbol, position))
            top += HEIGHT + (adjustable_height - HEIGHT)//2
            adjustable_height = HEIGHT
            text_detected = False
        else:
            top -= 1
            adjustable_height += 2
            text_detected = True
            if adjustable_height > max_adjustable_height:
                top += HEIGHT + (adjustable_height - HEIGHT)//2
                adjustable_height = HEIGHT
                text_detected = False

    if matches:
        for match in matches:
            if fuzz.partial_ratio(row['exchange'], match[0]) >= 80:
                return match
        for match in matches:
            if fuzz.partial_ratio(row['primaryExchange'], match[0]) >= 80:
                return match
                
        valid_exchanges = row['validExchanges'].split(',') if row['validExchanges'] else []
        for match in matches:
            for valid_exchange in valid_exchanges:
                if fuzz.partial_ratio(valid_exchange.strip(), match[0]) >= 80:
                    return match

def prepare_search_results(buffer, width=790, seconds=8):
    start = datetime.now()
    timeout = timedelta(seconds=seconds)
    i = 0
    while datetime.now() - start < timeout:
        try:
            search = pyautogui.locateOnScreen(f'assets/fund_setup/search{i}.png', confidence=0.8)
            break
        except Exception:
            i = (i + 1) % 3
    left = search.left
    top = search.top + search.height - buffer
    try:
        search_bottom = pyautogui.locateOnScreen(f'assets/fund_setup/search_bottom.png', confidence=0.9)
        height = (search_bottom.top - top) + search_bottom.height + (buffer - 21)
    except Exception:
        height = 300

    next_row = buffer + 1
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot_array = np.array(screenshot)
    if screenshot_array[next_row][-1].tolist() == [64, 64, 64, 255]:
        return screenshot, left, top, buffer
    elif screenshot_array[next_row][-1].tolist() == [109, 111, 113, 255]:
        raise Exception('search_etf() did not find the correct symbol')
    else:
        return prepare_search_results(buffer, seconds=seconds)

def search_eft(ocr, row, wait_time=5):
    scroll(999)
    pyautogui.press("esc")
    pyautogui.click((1880,100), interval=0.2)
    pyautogui.click(positions['search_box'], interval=0.2)
    pyautogui.write(row['longName'])
    pyautogui.press("enter")

    buffer = 15
    screenshot, left, top, buffer = prepare_search_results(buffer, seconds=wait_time)
    
    exchange, symbol, search_result = check_search_results(ocr, row, screenshot, left, top, buffer + 1)
    if search_result:
        pyautogui.click(search_result)
        pyautogui.press("enter", presses=3, interval=0.3)
        if select_exchange():
            exchange_bug = True
        else:
            exchange_bug = False
        clear_modals()
        return exchange, symbol, exchange_bug
    else:
        clear_modals()
        raise Exception('search_etf() did not find the correct symbol')

def quick_search_etf(row, count=None, name=None):
    pyautogui.press("esc")
    pyautogui.click((1880,100), interval=0.2)
    pyautogui.click(positions['search_box'], interval=0.2)
    if count:
        pyautogui.press("delete", presses=count)
        pyautogui.press("backspace", presses=count)

    if name:
        pyautogui.write(name)
    else:
        pyautogui.write(row['symbol'])
    pyautogui.press("enter", presses=3, interval=0.3)
    if select_exchange():
        exchange_bug = True
    else:
        exchange_bug = False
    clear_modals()
    return exchange_bug

# Overview functions
def process_profile(text_list):
    headings = ['TotalExpenseRatio', 'TotalNetAssets', 'BenchmarkIndex', 'Domicile', 'MarketGeoFocus', 'MarketCapFocus', 'FundCategory']
    current_label, current_values, labels, values = None, [], [], []
    threshold = 80

    for item in text_list:
        matches = [(heading, fuzz.partial_ratio(item, heading)) for heading in headings]
        best_match = max(matches, key=lambda x: x[1])

        if best_match[1] >= threshold and (current_label != best_match[0]):
            if current_label:
                labels.append(current_label)
                values.append(' '.join(current_values))
                current_values = []
            current_label = best_match[0]
        else:
            current_values.append(item)

    if current_label:
        labels.append(current_label)
        values.append(' '.join(current_values))

    return list(zip(labels, values))

def extract_profile(ocr):
    profile = pyautogui.locateOnScreen("assets/overview/profile.png", confidence=0.9)
    left = profile.left
    top = profile.top + profile.height
    lipper = pyautogui.locateOnScreen("assets/overview/lipper.png", confidence=0.9)
    width = 600
    height = lipper.top - top

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)

    text_list = extract_text(ocr, screenshot)
    scroll(-height/8)
    if text_list:
        return process_profile(text_list)
    else:
        raise Exception('skip')

def check_title(ocr, title, seconds=5, interval=1):
    start = datetime.now()
    timeout = timedelta(seconds=seconds)
    sleep(interval)
    while datetime.now() - start < timeout:

        screenshot = pyautogui.screenshot(region=(25, 100, 65, 25))
        screenshot = np.array(screenshot)
        text_color = (247, 247, 247, 255)
        if np.any(np.all(screenshot == text_color, axis=-1)):
            
            screenshot = pyautogui.screenshot(region=(25, 100, 480, 60))
            screenshot = np.array(screenshot)
            text = extract_text(ocr, screenshot)
            if text[0].upper().startswith(title):
                return text[0]
            else:
                return
        sleep(interval)

def check_tradable(seconds=5, interval=0.5):
    timeout = timedelta(seconds=seconds)
    start = datetime.now()
    sleep(interval)
    while datetime.now() - start < timeout:

        screenshot = pyautogui.screenshot(region=(25, 100, 65, 25))
        screenshot = np.array(screenshot)
        text_color = (247, 247, 247, 255)
        if np.any(np.all(screenshot == text_color, axis=-1)):

            screenshot = pyautogui.screenshot(region=(25, 125, 200, 40))
            screenshot = np.array(screenshot)
            nt_sign_color = (240, 71, 80, 255)
            if np.any(np.all(screenshot == nt_sign_color, axis=-1)):
                return False
            else:
                return True

def process_holding_types(text_list):
    for i, element in enumerate(text_list):
        if element.strip().isupper():
            text_list = text_list[:i]
            break

    last_label, labels , values = None, [], []
    for item in (text_list):
        if is_numerical(item):
            labels.append(last_label)
            values.append(item)
            last_label = None
        else:
            if last_label:
                labels.append(last_label)
                values.append(None)
            last_label = item

    return list(zip(labels, values))

def extract_holding_types(ocr):
    holdings = pyautogui.locateOnScreen("assets/overview/holdings.png", confidence=0.9)
    left = holdings.left
    top = holdings.top + holdings.height
    dividends = pyautogui.locateOnScreen("assets/overview/dividends.png", confidence=0.9)
    width = 600
    height = dividends.top - top

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)

    text_list = extract_text(ocr, screenshot)
    scroll(-height / 8)
    if len(text_list) > 1:
        return process_holding_types(text_list)
    else:
        raise Exception('skip')

def process_dividends(text_list):
    labels, values, value_indicator = [], [], False
    
    for item in text_list:
        if item[-1] == '%':
            value_indicator = True
        if value_indicator:
            values.append(item)
        else:
            labels.append(item)

    return list(zip(labels, values))

def extract_dividends(ocr):
    try:
        dividends = pyautogui.locateOnScreen("assets/overview/dividends.png", confidence=0.9)
        left = dividends.left
        top = dividends.top + dividends.height
        width = 600
        height = 200

        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        screenshot = np.array(screenshot)

        text_list = extract_text(ocr, screenshot)
        if 'IndustryAverage' not in text_list:
            return process_dividends(text_list)
    except Exception as e:
        raise Exception(f'extract_dividends() {e}')

def extract_style():
    style = pyautogui.locateOnScreen("assets/overview/style_matrix.png", confidence=0.9)
    left = style.left + style.width + 70
    top = style.top + style.height + 39
    width = 280
    height = 172

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)
    highlight_color = (29, 51, 88, 255)

    if np.any(np.all(screenshot == highlight_color, axis=-1)):
        styles = []
        rows = ['large', 'multi', 'mid', 'small']
        columns = ['value', 'core', 'growth']
        for i, row in enumerate(rows):
            row_step_px = round(height / (len(rows) - 1)) - 1
            for j, col in enumerate(columns):
                col_step_px = round(width / (len(columns) - 1)) - 1
                pixel = screenshot[row_step_px * i][col_step_px * j].tolist()
                styles.append((f'{row}-{col}', pixel == list(highlight_color)))
    
        return styles

def process_lipper(text_index, screenshot, width, height):
    bg_color = [24, 24, 24, 255]
    missing_color = [0, 0, 0, 255]
    row_idx = 16 + round(35 * text_index)
    for j in range(5):
        col_step_px = round(width/5) - 1
        pixel = screenshot[row_idx][col_step_px * (j+1)].tolist()
        if pixel == bg_color or pixel == missing_color:
            return j + 1

def extract_lipper(ocr):
    try:
        lipper = pyautogui.locateOnScreen("assets/overview/lipper.png", confidence=0.9)
        left = lipper.left
        top = lipper.top + lipper.height
        holdings = pyautogui.locateOnScreen("assets/overview/holdings.png", confidence=0.9)
        lipper_width = 300
        height = holdings.top - top

        screenshot = pyautogui.screenshot(region=(left, top, lipper_width, height))
        screenshot = np.array(screenshot)
        text_list = extract_text(ocr, screenshot)
        if text_list:
            width = 285
            screenshot = pyautogui.screenshot(region=(left+lipper_width+24, top+12, width, height-12))
            screenshot = np.array(screenshot)
            
            lipper = []
            for i, label in enumerate(text_list):
                value = process_lipper(i, screenshot, width, 34)
                lipper.append((label, value))
            scroll(-height/10)
            return lipper
        scroll(-height/10)
    except Exception as e:
        raise Exception(f'extract_dividends() {e}')
    
# Holding functions
def show_more(type=1):
    try:
        if type != 1:
            show_more = pyautogui.locateCenterOnScreen("assets/holdings/show_more2.png", confidence=0.9)
        else:
            show_more = pyautogui.locateCenterOnScreen("assets/holdings/show_more.png", confidence=0.9)
        pyautogui.click(show_more)
    except Exception:
        pass

def process_top10(text_list):
    # Assumes text is identified from left to right, and top to bottom
    index, current_labels, labels , values = True, [], [], []

    for item in text_list:
        if index and len(item) <= 2 and not item.endswith('%'):
            index = False 
        elif is_numerical(item) and item.endswith('%'):
            labels.append('-'.join(current_labels))
            values.append(item)
            current_labels = []
            index = True
        else:
            current_labels.append(item)
            index = False 
    return list(zip(labels, values))

def extract_top10(ocr):    
    top10 = pyautogui.locateOnScreen("assets/holdings/top10.png", confidence=0.9)
    left = top10.left
    top = top10.top + top10.height
    width = 626
    height = 455

    return capture_text(ocr, process_top10, left, top, width, height)

def process_industry(text_list):
    last_value, labels , values = None, [], []
    for item in (text_list):
        if is_numerical(item):
            if last_value:
                labels.append(None)
                values.append(last_value)
            last_value = item
        else:
            labels.append(item)
            values.append(last_value)
            last_value = None

    return list(zip(labels, values))

def process_holding_tables(text_list):
    last_value, labels , values = None, [], []
    for item in (text_list):
        if is_numerical(item):
            labels.append(last_value)
            values.append(item)
            last_value = None
        else:
            if last_value:
                labels.append(last_value)
                values.append(None)
            last_value = item
    return list(zip(labels, values))

def extract_industry(ocr):
    show_more()
    industry = pyautogui.locateOnScreen("assets/holdings/industry.png", confidence=0.9)
    scroll(-(industry.height*3/4))

    industry = pyautogui.locateOnScreen("assets/holdings/industry.png", confidence=0.9)
    left = industry.left + 40
    top = industry.top + industry.height
    width = 550
    try:
        show_less = pyautogui.locateOnScreen("assets/holdings/show_less.png", confidence=0.9)
        height = show_less.top - top
    except Exception:
        height = 450

    return capture_text(ocr, process_industry, left, top, width, height)

def extract_country(ocr):
    country = pyautogui.locateOnScreen("assets/holdings/country.png", confidence=0.9)
    scroll(-(country.top/15))
    show_more(2)
    country = pyautogui.locateOnScreen("assets/holdings/country.png", confidence=0.9)
    left = country.left + 50
    top = country.top + country.height
    width = 460
    currency = pyautogui.locateOnScreen("assets/holdings/currency.png", confidence=0.9)
    height = currency.top - top

    return capture_text(ocr, process_holding_tables, left, top, width, height)

def extract_currency(ocr):
    currency = pyautogui.locateOnScreen("assets/holdings/currency.png", confidence=0.9)
    scroll(-(currency.top/20))
    show_more(2)
    scroll(-(currency.top/50), 1)
    currency = pyautogui.locateOnScreen("assets/holdings/currency.png", confidence=0.9)
    left = currency.left + 50
    top = currency.top + currency.height
    width = 460
    try:
        debtor = pyautogui.locateOnScreen("assets/holdings/debtor_quality.png", confidence=0.9)
        height = debtor.top - top
    except Exception:
        height = BOTTOM - top

    return capture_text(ocr, process_holding_tables, left, top, width, height)

def capture_text(ocr, function, left, top, width, height):
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)
    text_list = extract_text(ocr, screenshot)
    if text_list:
        try:
            return function(text_list)
        except:
            expand_px = 4
            screenshot = screenshot = pyautogui.screenshot(region=(left - expand_px, top - expand_px, width + expand_px*2, height + expand_px*2))
            screenshot = np.array(screenshot)
            text_list = extract_text(ocr, screenshot)
            if text_list:
                return function(text_list)
            
# Bond functions
def extract_debtors(ocr, name):
    scroll(-99, 1)
    debtor = pyautogui.locateOnScreen(f"assets/holdings/{name}.png", confidence=0.9)
    left = debtor.left
    top = debtor.top + debtor.height
    width = 405
    height = BOTTOM - top

    return capture_text(ocr, process_holding_tables, left, top, width, height)

# Fundamentals functions
def extract_fundamentals_text(ocr, screenshot):
    # display(Image.fromarray(screenshot))
    try:
        results = ocr.ocr(screenshot, cls=False)
    except Exception as e:
        print(f"Error during OCR: {e}")
        results = [None]

    text_list = []
    if results and results[0]:
        for res in results:
            for line in res:
                bbox, (text, conf) = line[0], line[-1]
                if conf > 0.85:
                    text_list.append({
                        'text': text.replace(' ', ''),
                        'bbox': bbox,
                        'conf': conf
                    })
    return text_list

def calculate_value_crop(label_bbox, screenshot, offset=5, expansion=0):
    """
    Given a label's bounding box, calculates a region where its value should be.
    The region is expanded by 'expansion' pixels on all sides.
    """
    # Extract coordinates from label_bbox (assumed to be list of 4 points)
    x_coords = [pt[0] for pt in label_bbox]
    y_coords = [pt[1] for pt in label_bbox]
    label_left, label_top = min(x_coords), min(y_coords)
    label_right, label_bottom = max(x_coords), max(y_coords)
    
    # Initial value region: to the right of the label with a small offset.
    initial_value_left = label_right + offset
    initial_value_top = label_top
    screenshot_width = screenshot.shape[1]
    initial_value_width = screenshot_width - initial_value_left
    initial_value_height = label_bottom - label_top

    # Expand the region by 'expansion' pixels on all sides.
    new_x = max(initial_value_left, 0)
    new_y = max(initial_value_top - expansion, 0)
    new_width = initial_value_width
    new_height = initial_value_height + 2 * expansion

    # Ensure the region stays within the screenshot boundaries.
    if new_x + new_width > screenshot.shape[1]:
        new_width = screenshot.shape[1] - new_x
    if new_y + new_height > screenshot.shape[0]:
        new_height = screenshot.shape[0] - new_y

    return (int(new_x), int(new_y), int(new_width), int(new_height))

def detect_value_with_expansion(ocr, screenshot, label_bbox, initial_offset=5, max_expansion=49):
    """
    Attempts to detect a value by progressively expanding the crop region.
    Returns the first detected text or None if no detection is made within max_expansion.
    """
    expansion = 0
    new_det = []
    while not new_det and expansion <= max_expansion:
        crop_region = calculate_value_crop(label_bbox, screenshot, offset=initial_offset, expansion=expansion)
        cropped = screenshot[crop_region[1]:crop_region[1]+crop_region[3],
                             crop_region[0]:crop_region[0]+crop_region[2]]
        new_det = extract_fundamentals_text(ocr, cropped)
        if new_det:
            break
        expansion += 1
    return new_det[0]['text'] if new_det else None

def process_fundamentals(detections, screenshot, ocr):
    last_label, last_bbox = None, None
    labels, values = [], []
    
    for det in detections:
        text = det['text']
        bbox = det['bbox']
        if is_numerical(text) or text.isupper():
            if last_label is not None:
                labels.append(last_label)
                values.append(text)
                last_label = None  # reset after pairing
        else:
            if text == 'Equity':
                continue
            elif last_label:
                labels.append(last_label)
                new_value = detect_value_with_expansion(ocr, screenshot, last_bbox, initial_offset=5)
                values.append(new_value)
            last_label = text
            last_bbox = bbox
    
    # Handle a leftover label.
    if last_label and last_label != 'Equity':
        labels.append(last_label)
        new_value = detect_value_with_expansion(ocr, screenshot, last_bbox, initial_offset=5)
        values.append(new_value)

    return list(zip(labels, values))

def extract_fundamentals(ocr, prev_list=None):
    try:
        top_screenshot_boundary = pyautogui.locateOnScreen("assets/fundamentals/metric.png", confidence=0.9)
    except pyautogui.ImageNotFoundException:
        top_screenshot_boundary = pyautogui.locateOnScreen("assets/fundamentals/top_border.png", confidence=0.9)
    left = top_screenshot_boundary.left
    top = top_screenshot_boundary.top + top_screenshot_boundary.height
    width = 550
    height = BOTTOM - top

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)
    
    text_list = extract_fundamentals_text(ocr, screenshot)
    if text_list:
        if len(text_list) <= 15:
            return process_fundamentals(text_list, screenshot, ocr)
        else:
            current_list = process_fundamentals(text_list, screenshot, ocr)
            if prev_list:
                if set(current_list) == set(prev_list):
                    return current_list
                return list(set(current_list + prev_list))
            scroll(-999)
            return extract_fundamentals(ocr, current_list)
    raise Exception('extract_fundamentals() error')

# Cleaning functions
def clean_labels(label, col):
    if col == 'industries':
        if isinstance(label, str):
            if label.endswith('-Discontinuedeff09/19/2020'):
                return label.split('-')[0]
        return label
    
    elif col == 'holding_types':
        if isinstance(label, str):
            if label.startswith('■'):
                return label[1:]
            elif label.startswith('1'):
                return label[1:]
        return label
    elif col == 'debtors':
        if isinstance(label, str):
            if ('（') in label:
                return label.replace('（', '(')
        return label
    elif col == 'fundamentals':
        if isinstance(label, str):
            if label == 'LTDebt/ShareholdersEquity':
                return 'LTDebt/Shareholders'
        return label
    return label
    
def correct_digit(value_str):
    try:
        digit = re.sub(r'[^\d.-]', '', value_str).strip()
        return float(digit)
    except Exception:
        return value_str

def clean_values(value_str, col):
    # print(value_str)
    if col == 'profile':
        return value_str
    if isinstance(value_str, str):
        if value_str.endswith('%'):
            return correct_digit(value_str.replace('%',''))/100
        try:
            return correct_digit(value_str)
        except Exception:
            return value_str
    return value_str

def clean_df(df):
    for col in df.columns:
        # print(col)
        df[col] = df[col].apply(evaluate_literal)
        df[col] = df[col].apply(lambda x: [(clean_labels(item[0], col), item[1]) if isinstance(item, tuple) and len(item) == 2 else item for item in x] if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: [(item[0], clean_values(item[1], col)) if isinstance(item, tuple) and len(item) == 2 else item for item in x] if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: sorted(x, key=lambda item: item[0] if isinstance(item, tuple) and item[0] else '') if isinstance(x, list) else x)
    return df

# Prep functions
def evaluate_literal(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    
def load(path):
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].apply(evaluate_literal)
    return df

def save(df):
    final_df = df[df.apply(is_row_valid, axis=1)]
    final_df = clean_df(final_df)
    try:
        temp_df = load('data/contract_elaborated.csv')
        temp_df = clean_df(temp_df)
        final_df = pd.concat([final_df, temp_df]).drop_duplicates(subset=['symbol', 'exact_search', 'search_exchange', 'search_symbol'])
    except FileNotFoundError:
        pass

    # Filter out the duplicates with 'exact_search' is False
    duplicates_df = final_df[final_df.duplicated(subset='symbol', keep=False)]
    final_df = final_df.drop(duplicates_df[duplicates_df['exact_search'] == False].index)

    final_df.to_csv('data/contract_elaborated.csv', index=False)

def is_numerical(val):
    try:
        val = str(val).replace('%', '')
        float(val)
        return True
    except Exception:
        return False

def is_valid_tuple(tuple, column):
    def extract_float(value):
        match = re.match(r'[^0-9]*([0-9.,]+)', value)
        if match:
            return float(match.group(1).replace(',', ''))
        return None
    
    label, value = tuple
    if not isinstance(label, str): # keep
        # if label != None: # Comment out for more rigid filter
        return False
    if value is None:
        return True # Comment out for more rigid filter
        return False 
    if is_numerical(value):
        return True
    
    if column == 'profile':
        if value and label:
            return True
    if column == 'fundamentals':
        if value.isupper():
            return True
    if column == 'dividends':
        if value == 'Unknown':
            return True
        extract_float_value = extract_float(value)
        if extract_float_value is not None:
            return True
    if column == 'style':
        if isinstance(value, bool):
            return True
    return False

def is_row_valid(row):
    for col in row.index:
        if isinstance(row[col], list):
            # if col == 'fundamentals':
            #     if len(row[col]) not in [4,5,21,22,   23]: #4, 5, 21, 22 are the acceptable num of fund values, 23 is for little bugs
            #         print(len(row[col]))
            #         return False
            for tuple in row[col]:
                if not is_valid_tuple(tuple, col):
                    print(tuple)
                    return False
    return True

def has_bad_multiplier(long_name):
    cleaned = long_name.replace('-', '').replace('+', '')
    for word in cleaned.split():
        if re.fullmatch(r'\d+X', word):
            if int(word[:-1]) > 1:
                return True
    return False

def get_remaining():
    contract_details = load('data/contract_details.csv')
    try:
        final_df = load('data/contract_elaborated.csv')
        final_df = final_df[final_df.apply(is_row_valid, axis=1)]

        exclusion_condition = (final_df['exchange_bug'] == True) | (final_df['exact_search'] == True) | (~final_df['profile'].isna())
        # exclusion_condition = (final_df['exchange_bug'] == True) | (final_df['exact_search'] == True)
        symbols_to_exclude = final_df[exclusion_condition]['symbol']
        remaining = contract_details[~contract_details['symbol'].isin(symbols_to_exclude)]

        # # To debug invalid rows
        # remaining = final_df.copy()
        # remaining = remaining[~remaining.apply(is_row_valid, axis=1)]
    except FileNotFoundError:
        remaining = contract_details.copy()
        
    remaining = remaining[~remaining['longName'].apply(has_bad_multiplier)]
    remaining = remaining[['symbol', 'exchange', 'primaryExchange', 'validExchanges', 'currency', 'conId', 'longName', 'stockType', 'isin']]
    return remaining
