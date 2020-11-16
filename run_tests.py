from sys import argv
import environment_tests

if __name__ == '__main__':
    tests = argv[1]

    log_dict = {
            'Create datasets':{},
            'Train datasets':{},
            'Test datasets':{}
            }
    if tests in ['mnist', 'all']:
        log_dict = environment_tests.run_mnist(log_dict)
    
    if tests in ['spotify', 'all']:
        log_dict = environment_tests.run_spoti(log_dict)

    environment_tests.print_loop(log_dict)
