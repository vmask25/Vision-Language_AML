import os
import logging
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from load_data_dg import build_splits_baseline_dg, build_splits_domain_disentangle_dg, build_splits_clip_disentangle_dg
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment

def main(opt):
    # Setting up the experiments with the relative dataloaders, discriminating when using domain generalization
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline(opt) if not opt['dom_gen'] else build_splits_baseline_dg(opt)
    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        if not opt['dom_gen']:
            source_train_loader, target_train_loader, source_validation_loader, test_loader = build_splits_domain_disentangle(opt)
        else:    
            source_train_loader, source_validation_loader, test_loader = build_splits_domain_disentangle_dg(opt)
    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        if not opt['dom_gen']:
            source_train_loader, target_train_loader, source_descriptions_train_loader, target_descriptions_train_loader, source_validation_loader, test_loader = build_splits_clip_disentangle(opt)
        else:
            source_train_loader, source_validation_loader, test_loader, source_descriptions_train_loader= build_splits_clip_disentangle_dg(opt)
    else:
        raise ValueError('Experiment not yet supported.')

    # Setup logger
    if opt['experiment'] == 'domain_disentangle':
        alpha = opt['alpha']
        target_dom = opt['target_domain']
        logging.basicConfig(filename=f'{opt["output_path"]}/log_{target_dom}_w1={experiment.weights[0]}_w2={experiment.weights[1]}_w3={experiment.weights[2]}_alpha={alpha}.txt', format='%(message)s', level=logging.INFO, filemode='a')
    else:
        logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    if not opt['test']: # Skip training if '--test' flag is set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        # Restore last checkpoint, if present
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        if opt['experiment'] == 'baseline':
            # Train loop
            while iteration < opt['max_iterations']:
                for data in train_loader:
                    
                    total_train_loss += experiment.train_iteration(data)

                    if iteration % opt['print_every'] == 0:
                        logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(validation_loader)
                        logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, val_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

        elif opt['experiment'] == 'domain_disentangle':

            # Defining an iterator for the target data loader (when not using domain generalization)
            if not opt['dom_gen']:
                target_train_loader_iterator = iter(target_train_loader)

            # Train loop
            while iteration < opt['max_iterations']:

                for source_data in source_train_loader:
                    
                    # Getting the next target batch
                    if not opt['dom_gen']:
                        try:
                            target_data = next(target_train_loader_iterator)
                        except StopIteration:
                            # Restarting the iterator if the source loader is bigger
                            target_train_loader_iterator = iter(target_train_loader)
                            target_data = next(target_train_loader_iterator)
                    
                    total_train_loss += experiment.train_iteration(source_data, 0)
                    if not opt['dom_gen']:
                        total_train_loss += experiment.train_iteration(target_data, 1)

                    if iteration % opt['print_every'] == 0:
                        logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(source_validation_loader)
                        logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, val_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

        elif opt['experiment'] == 'clip_disentangle':

            # Defining iterators for the target data loader and target descriptions data loader (when not using domain generalization)
            if not opt['dom_gen']:
                target_train_loader_iterator = iter(target_train_loader)
                target_descriptions_train_loader_iterator = iter(target_descriptions_train_loader)
            # Defining an iterator for the source descriptions data loader
            source_descriptions_train_loader_iterator = iter(source_descriptions_train_loader)

            # Train loop
            while iteration < opt['max_iterations']:

                #for target_data in target_train_loader:
                for source_data in source_train_loader:

                    if not opt['dom_gen']:
                        try:
                            target_data = next(target_train_loader_iterator)
                        except StopIteration:
                            target_train_loader_iterator = iter(target_train_loader)
                            target_data = next(target_train_loader_iterator)

                        try:
                            target_descriptions_data = next(target_descriptions_train_loader_iterator)
                        except StopIteration:
                            target_descriptions_train_loader_iterator = iter(target_descriptions_train_loader)
                            target_descriptions_data = next(target_descriptions_train_loader_iterator)

                    try:
                        source_descriptions_data = next(source_descriptions_train_loader_iterator)
                    except StopIteration:
                        source_descriptions_train_loader_iterator = iter(source_descriptions_train_loader)
                        source_descriptions_data = next(source_descriptions_train_loader_iterator)
                     
                    # Each batch should follow a different path in the iteration
                    if len(source_data[0]) != 1:
                        total_train_loss += experiment.train_iteration(source_data, 0, 0)
                    if not opt['dom_gen'] and len(target_data[0]) != 1:
                        total_train_loss += experiment.train_iteration(target_data, 1, 0)
                    if len(source_descriptions_data[0]) != 1:
                        total_train_loss += experiment.train_iteration(source_descriptions_data, 0, 1)
                    if not opt['dom_gen'] and len(target_descriptions_data[0]) != 1:
                        total_train_loss += experiment.train_iteration(target_descriptions_data, 1, 1)

                    if iteration % opt['print_every'] == 0:
                        logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                    
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        val_accuracy, val_loss = experiment.validate(source_validation_loader)
                        logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, val_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

        else:
            raise ValueError('Experiment not yet supported.')
        

    # Test1 using the best checkpoint
    iteration, best_accuracy, _ = experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST with BEST] Accuracy: {(100 * test_accuracy):.2f} \n(best accuracy {(100 * best_accuracy):.2f} registered at {iteration})')

    if opt['experiment'] == 'domain_disentangle':
        logging.info(f'Tested with weights: w1={experiment.weights[0]}, w2={experiment.weights[1]}, w3={experiment.weights[2]}, alpha={alpha}')

    # Test2 using the last checkpoint
    iteration, last_accuracy, _ = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST with LAST] Accuracy: {(100 * test_accuracy):.2f} \n(last accuracy {(100 * last_accuracy):.2f} registered at {iteration})')

    if opt['experiment'] == 'domain_disentangle':
        logging.info(f'Tested with weights: w1={experiment.weights[0]}, w2={experiment.weights[1]}, w3={experiment.weights[2]}, alpha={alpha}')

if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    main(opt)
