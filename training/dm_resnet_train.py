from sklearn.cross_validation import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
import os, argparse
from meta import UNIMAGED_INT, DMMetaManager
from dm_image import DMImageDataGenerator
from dm_resnet import ResNetBuilder

def run(img_folder, img_extension='png', img_size=[288, 224], 
        batch_size=16, samples_per_epoch=160, nb_epoch=20, weight_decay=.0001,
        val_size=.2, lr_patience=5, net='resnet50', nb_worker=4,
        exam_tsv='./metadata/exams_metadata.tsv',
        img_tsv='./metadata/images_crosswalk.tsv',
        trained_model='./modelState/dm_resnet_model.h5'):

    # Setup training and validation data.
    random_seed = os.getenv('RANDOM_SEED', 12345)
    meta_man = DMMetaManager(exam_tsv=exam_tsv, img_tsv=img_tsv, 
                             img_folder=img_folder, img_extension=img_extension)
    img_list, lab_list = meta_man.get_flatten_img_list()
    img_train, img_val, lab_train, lab_val = train_test_split(
        img_list, lab_list, test_size=val_size, random_state=random_seed, stratify=lab_list)
    img_gen = DMImageDataGenerator(
        samplewise_center=True, 
        samplewise_std_normalization=True, 
        horizontal_flip=True, 
        vertical_flip=True)
    train_generator = img_gen.flow_from_img_list(
        img_train, lab_train, target_size=(img_size[0], img_size[1]), 
        batch_size=batch_size, seed=random_seed)
    val_generator = img_gen.flow_from_img_list(
        img_val, lab_val, target_size=(img_size[0], img_size[1]), 
        batch_size=batch_size, seed=random_seed)

    # Model training.
    if net == 'resnet18':
        model = ResNetBuilder.build_resnet_18(
            (1, img_size[0], img_size[1]), 1, weight_decay)
    elif net == 'resnet34':
        model = ResNetBuilder.build_resnet_34(
            (1, img_size[0], img_size[1]), 1, weight_decay)
    elif net == 'resnet50':
        model = ResNetBuilder.build_resnet_50(
            (1, img_size[0], img_size[1]), 1, weight_decay)
    elif net == 'resnet59':
        model = ResNetBuilder.build_dm_resnet_59(
            (1, img_size[0], img_size[1]), 1, weight_decay)
    elif net == 'resnet68':
        model = ResNetBuilder.build_dm_resnet_68(
            (1, img_size[0], img_size[1]), 1, weight_decay)
    elif net == 'resnet101':
        model = ResNetBuilder.build_resnet_101(
            (1, img_size[0], img_size[1]), 1, weight_decay)
    elif net == 'resnet152':
        model = ResNetBuilder.build_resnet_152(
            (1, img_size[0], img_size[1]), 1, weight_decay)

    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', 
                  metrics=['accuracy', 'precision', 'recall'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=lr_patience, verbose=1)
    model.fit_generator(train_generator, 
                        samples_per_epoch=samples_per_epoch, 
                        nb_epoch=nb_epoch, 
                        validation_data=val_generator, 
                        nb_val_samples=len(img_val), 
                        callbacks=[reduce_lr], 
                        nb_worker=nb_worker, 
                        pickle_safe=True  # turn on pickle_safe to avoid a strange error.
                        )
    model.save(trained_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM ResNet training")
    parser.add_argument("img_folder", type=str)
    parser.add_argument("--img-extension", "-ext", dest="img_extension", 
                        type=str, default="png")
    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int, 
                        default=[288, 224])
    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", "-spe", dest="samples_per_epoch", 
                        type=int, default=160)
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=20)
    parser.add_argument("--weight-decay", "-wd", dest="weight_decay", 
                        type=float, default=.0001)
    parser.add_argument("--val-size", "-vs", dest="val_size", type=float, default=.2)
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=5)
    parser.add_argument("--net", dest="net", type=str, default="resnet50")
    parser.add_argument("--nb-worker", "-nw", dest="nb_worker", type=int, default=4)
    parser.add_argument("--exam-tsv", "-et", dest="exam_tsv", type=str, 
                        default="./metadata/exams_metadata.tsv")
    parser.add_argument("--img-tsv", "-it", dest="img_tsv", type=str, 
                        default="./metadata/images_crosswalk.tsv")
    parser.add_argument("--trained-model", "-m", dest="trained_model", type=str, 
                        default="./modelState/dm_resnet_model.h5")

    args = parser.parse_args()
    run(args.img_folder, 
        img_extension=args.img_extension, 
        img_size=args.img_size, 
        batch_size=args.batch_size, 
        samples_per_epoch=args.samples_per_epoch, 
        nb_epoch=args.nb_epoch, 
        weight_decay=args.weight_decay,
        val_size=args.val_size, 
        lr_patience=args.lr_patience, 
        net=args.net,
        nb_worker=args.nb_worker,
        exam_tsv=args.exam_tsv,
        img_tsv=args.img_tsv,
        trained_model=args.trained_model
        )


