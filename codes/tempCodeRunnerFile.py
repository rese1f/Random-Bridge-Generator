    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epoch,)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)