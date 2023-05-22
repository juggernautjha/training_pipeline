#!/usr/bin/env python3
import os
import json
from keras_cv_attention_models.imagenet import data, train_func, losses


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_arguments(file):
    import json
    file = open(file, 'r')
    # print(file.read())
    args = json.load(file)
    args = dotdict(args)
    # args.additional_model_kwargs = {"drop_connect_rate": 0.05}
    args.additional_model_kwargs = json.loads(args.additional_model_kwargs) if args.additional_model_kwargs else {}

    lr_decay_steps = args.lr_decay_steps.strip().split(",")
    if len(lr_decay_steps) > 1:
        # Constant decay steps
        args.lr_decay_steps = [int(ii.strip()) for ii in lr_decay_steps if len(ii.strip()) > 0]
    else:
        # Cosine decay
        args.lr_decay_steps = int(lr_decay_steps[0].strip())

    if args.basic_save_name is None and args.restore_path is not None:
        basic_save_name = os.path.splitext(os.path.basename(args.restore_path))[0]
        basic_save_name = basic_save_name[:-7] if basic_save_name.endswith("_latest") else basic_save_name
        args.basic_save_name = basic_save_name
    elif args.basic_save_name is None or args.basic_save_name.startswith("_"):
        data_name = args.data_name.replace("/", "_")
        basic_save_name = "{}_{}_{}_{}_batchsize_{}".format(args.model, args.input_shape, args.optimizer, data_name, args.batch_size)
        basic_save_name += "_randaug_{}_mixup_{}_cutmix_{}_RRC_{}".format(args.magnitude, args.mixup_alpha, args.cutmix_alpha, args.random_crop_min)
        basic_save_name += "_lr512_{}_wd_{}".format(args.lr_base_512, args.weight_decay)
        args.basic_save_name = basic_save_name if args.basic_save_name is None else (basic_save_name + args.basic_save_name)
    args.enable_float16 = not args.disable_float16
    args.tensorboard_logs = None if args.tensorboard_logs is None or args.tensorboard_logs.lower() == "none" else args.tensorboard_logs

    return args


# Wrapper this for reuse in progressive_train_script.py
def run_training_by_args(args):
    print(">>>> ALl args:", args)
    # return None, None, None

    strategy = train_func.init_global_strategy(args.enable_float16, args.seed, args.TPU)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    input_shape = (args.input_shape, args.input_shape)
    use_token_label = False if args.token_label_file is None else True
    use_teacher_model = False if args.teacher_model is None else True
    teacher_model_input_shape = input_shape if args.teacher_model_input_shape == -1 else (args.teacher_model_input_shape, args.teacher_model_input_shape)

    # Init model first, for in case of use_token_label, getting token_label_target_patches
    total_images, num_classes, steps_per_epoch, num_channels = data.init_dataset(args.data_name, batch_size=batch_size, info_only=True)
    input_shape = (*input_shape, num_channels)  # Just in case channel is not 3, like mnist being 1...
    teacher_model_input_shape = (*teacher_model_input_shape, num_channels)  # Just in case channel is not 3, like mnist being 1...
    assert not (num_channels != 3 and args.rescale_mode == "torch")  # "torch" mode mean and std are 3 channels
    with strategy.scope():
        model = args.model if args.restore_path is None else args.restore_path
        model = train_func.init_model(model, input_shape, num_classes, args.pretrained, **args.additional_model_kwargs)
        model = train_func.model_post_process(model, args.freeze_backbone, args.freeze_norm_layers, use_token_label)
        if args.summary:
            model.summary()

        if use_teacher_model:
            print(">>>> [Build teacher model]")
            teacher_model = train_func.init_model(
                args.teacher_model, teacher_model_input_shape, num_classes, args.teacher_model_pretrained, reload_compile=False
            )
            model, teacher_model = train_func.init_distill_model(model, teacher_model)
        else:
            teacher_model = None
    token_label_target_patches = model.output_shape[-1][1:-1] if use_token_label else -1

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=args.data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        rescale_mode=args.rescale_mode,
        eval_central_crop=args.eval_central_crop,
        random_crop_min=args.random_crop_min,
        resize_method=args.resize_method,
        resize_antialias=not args.disable_antialias,
        random_erasing_prob=args.random_erasing_prob,
        magnitude=args.magnitude,
        num_layers=args.num_layers,
        use_positional_related_ops=not args.disable_positional_related_ops,
        token_label_file=args.token_label_file,
        token_label_target_patches=token_label_target_patches,
        teacher_model=teacher_model,
        teacher_model_input_shape=teacher_model_input_shape,
    )

    lr_base = args.lr_base_512 * batch_size / 512
    warmup_steps, cooldown_steps, t_mul, m_mul = args.lr_warmup_steps, args.lr_cooldown_steps, args.lr_t_mul, args.lr_m_mul  # Save line-width
    lr_scheduler, lr_total_epochs = train_func.init_lr_scheduler(
        lr_base, args.lr_decay_steps, args.lr_min, args.lr_decay_on_batch, args.lr_warmup, warmup_steps, cooldown_steps, t_mul, m_mul
    )
    epochs = args.epochs if args.epochs != -1 else lr_total_epochs

    with strategy.scope():
        token_label_loss_weight = args.token_label_loss_weight if use_token_label else 0
        distill_loss_weight = args.distill_loss_weight if use_teacher_model else 0
        loss, loss_weights, metrics = train_func.init_loss(
            args.bce_threshold, args.label_smoothing, token_label_loss_weight, distill_loss_weight, args.distill_temperature, model.output_names
        )

        if model.optimizer is None:
            # optimizer can be a str like "sgd" / "adamw" / "lamb", or specific initialized `keras.optimizers.xxx` instance.
            # Or just call `model.compile(...)` by self.
            model = train_func.compile_model(model, args.optimizer, lr_base, args.weight_decay, loss, loss_weights, metrics, args.momentum)
        print(">>>> basic_save_name =", args.basic_save_name)
        # return None, None, None
        latest_save, hist = train_func.train(
            model, epochs, train_dataset, test_dataset, args.initial_epoch, lr_scheduler, args.basic_save_name, logs=args.tensorboard_logs
        )
    return model, latest_save, hist


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])
    run_training_by_args(args)
