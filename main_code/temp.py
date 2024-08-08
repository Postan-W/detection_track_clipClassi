from colorama import init, Fore, Back, Style
import time
def colored_progress_bar():
    init(autoreset=True)
    for i in range(101):
        percentage = Fore.GREEN + f'{i}%' + Fore.RESET
        filled_length = int(i // 2)
        bar = Back.WHITE + Fore.BLUE + '#' * filled_length + Fore.RESET + Back.RESET + '.' * (50 - filled_length)
        print(f'\r{bar} {percentage}', end='')
        time.sleep(0.1)

if __name__ == '__main__':
    colored_progress_bar()

