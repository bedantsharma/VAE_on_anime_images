# VAE_on_anime_images
## data set used 
[dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
i used 20000 images form this dataset to train my model although this data set contains some 60000+ images 
## Model -
### inupt 
the model takes 64*64*3 jpg images as input in form of a numpy array 
### encoder - 
the model consists an encoder which is made up by various convolutional layers i also added maxpooling layer somewhere in between 
### sampling Z - 
encoder produces two varibles mu and sigma that are then used to resample a Z by using the Reparameterization trick if you want to know more about that you can reffer this [link](https://www.tensorflow.org/tutorials/generative/cvae)
### decoder - 
the decoder is nothing but mirror image of the encder used.

### loss function - 
in total two loss funciton are used one is kl divergence loss and other is reconstruction loss which is noting but the mean squared difference between the generated and the input image 
code is hilighted here-
```
            z_mean, z_log_var, reconstruction = self(data)
            beta = 500
            reconstruction_loss = tf.reduce_mean(
                beta
                * losses.mean_squared_error(
                    data, reconstruction
                )
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5
                    * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss
```
