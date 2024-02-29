# GAN_Face_Anonymity
代码copy自https://github.com/chi0tzp/FALCO

Authors official PyTorch implementation of the Attribute-preserving Face Dataset Anonymization via Latent Code Optimization (CVPR 2023, top-10%). If you find this code useful for your research, please cite our paper.

    Attribute-preserving Face Dataset Anonymization via Latent Code Optimization
    Simone Barattin*, Christos Tzelepis*, Ioannis Patras, and Nicu Sebe (* denotes co-first authorship)
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2023, highlight/top-10%)

    Abstract: This work addresses the problem of anonymizing the identity of faces in a dataset of images, such that the privacy of those depicted is not violated, while at the same time the dataset is useful for downstream task such as for training machine learning models. To the best of our knowledge, we are the first to explicitly address this issue and deal with two major drawbacks of the existing state-of-the-art approaches, namely that they (i) require the costly training of additional, purpose-trained neural networks, and/or (ii) fail to retain the facial attributes of the original images in the anonymized counterparts, the preservation of which is of paramount importance for their use in downstream tasks. We accordingly present a task-agnostic anonymization procedure that directly optimises the images' latent representation in the latent space of a \textit{pre-trained} GAN. By optimizing the latent codes directly, we ensure both that the identity is of a desired distance away from the original (with an identity obfuscation loss), whilst preserving the facial attributes (using a novel feature-matching loss in FaRL's deep feature space). We demonstrate through a series of both qualitative and quantitative experiments that our method is capable of anonymizing the identity of the images whilst--crucially--better-preserving the facial attributes.

摘要：这项工作旨在解决图像数据集中人脸身份匿名化的问题，从而使用户人脸隐私不会受到泄漏，同时该数据集对于下游任务例如训练机器学习模型等也很有用。论文解决两个主要问题，即（i）需要额外昂贵的、专门训练的神经网络；（ii）无法在匿名对应图像中保留原始图像的面部属性，其保留对于它们在下游任务中的使用至关重要。因此，我们提出了一种与任务无关的匿名化过程，可以直接优化预训练GAN潜在空间中图像的潜在表示。通过直接优化潜在代码，我们确保身份与原始身份具有所需的距离（具有身份混淆损失），同时保留面部属性（在FaRL[49]深度特征空间中使用新颖的特征匹配损失）。通过一系列定性和定量实验证明，方法能够匿名化图像的身份同时更好地保留面部属性。

原论文方法的整个匿名化过程包括了GAN逆向（e4e），FaRL ViT编码器，以及StyleGAN2生成器等多个组件。最终实现在确保隐私保护的同时，创建一个在属性上与原始数据集保持高度一致的匿名化数据集。
	真实数据集X_R：包含真实人脸图像的数据集，希望通过这个匿名化过程来保护这些图像中个人的隐私。
	GAN逆向(e4e)：这个步骤涉及到使用一个称为e4e的预训练模型，它能够将真实的人脸图像映射到一个潜在空间中。在这个潜在空间中，图像被表示为一个潜码W_R^+。
	 FaRL ViT 编码器（FaRL ViT Encoder）：这个编码器接收真实数据集X_R和伪造数据集X_F中的图片作为输入，然后在特征空间中对它们进行编码。这个特征空间捕捉了人脸图像的关键特征，如面部结构、表情等。这里的X_F代表伪造（或生成）的图像数据集，它的生成和作用可以这样理解：X_F从正态分布Z\sim N\left(0,I\right)中采样潜在向量Z\ 。使用生成器G通过潜在向量Z生成伪造图像X_F。生成器G是一个预训练的 StyleGAN2 模型。生成的伪造数据集X_F的规模\left|X_F\right|大于原始真实数据集X_R的规模。X_F用于增强和扩充原始数据集X_R，提供更多的数据样本以便于训练和优化过程。在某些情况下，X_F可以作为对抗样本，帮助编码器\mathcal{E}_\mathcal{F}学习区分真实和伪造的图像特征，从而提高特征提取的准确性和鲁棒性。FaRL ViT Encoder使用这两个数据集来提取和比较特征，确保生成的匿名化图像X_A在保留关键属性的同时，其身份与真实图像X_R的身份不同，实现隐私保护。
	FaRL 特征空间：在这个空间中，每个图像都有其相应的特征表示，每个图像都被编码为一组特征向量。这些特征向量捕获了图像的重要视觉信息，如面部特征等。这个表示可以用于后续的优化步骤，以确保在匿名化过程中保留图像的关键属性。
	k-NN: 在FaRL Feature Space中，使用k-NN (k-Nearest Neighbors) 算法来寻找最近的特征向量对。这些对应关系有助于在优化过程中保持关键的面部属性。
	属性保持和身份混淆优化：这个过程保证了生成的图像在保留原始图像属性的同时，隐藏了个人身份。通过 L_{att} 保持关键面部属性，这个损失函数确保在匿名化后的图像中仍然保留了原图像的关键属性。通过L_{id}进行身份混淆，这个损失函数确保生成的图像在身份上与原图像有所区别，以保护个人隐私。
	StyleGAN2 生成器 (G)：这个生成器接收修改后的潜码W_A^+，并生成匿名化后的人脸图像X_A。生成器被训练来根据潜码生成图像，同时还被优化以保留图像的关键属性并混淆身份。
	匿名化数据集 X_A：这是最终产生的数据集，它包含了匿名化后的人脸图像。这些图像在视觉上与真实的人脸图像相似，但已经进行了修改，以便不能被用来识别出图像中的个人。
\mathbit{W}_\mathbit{F}^\mathbit{i}和\mathbit{W}_\mathbit{F}^+的关系：整个流程的核心是通过潜码优化来在保留关键属性的同时实现面部匿名化。这通过调整潜码中可训练的部分W_A^+来实现，以便生成的图像与真实的原图在视觉属性上是一致的，但在个人身份上不同。W_F^+是通过从StyleGAN2的\mathcal{Z}潜在空间采样得到的潜码，这个潜在空间遵循标准高斯分布N\left(0,I\right)。采样得到的\mathcal{Z}潜码之后被转换（通过输入MLP，即多层感知机）为W^+潜码。这些W^+潜码可以被用来通过StyleGAN2生成器G生成伪造的人脸图像X_F。同时，真实数据集X_R中的图像通过e4e编码器被转换为相应的W_R^+潜码。这个过程将真实图像对应到StyleGAN2的潜在空间，从而可以使用生成器G重构出与原始真实图像相似的图像。在配对真实图像X_R和伪造图像X_F的过程中，使用了一个基于FaRL的ViT（VisionTransformer）编码器\mathcal{E}_\mathcal{F}来表示每个数据集中所有图像的特征。基于这些特征表示，使用k-NN分类器来找出与每个真实图像在欧几里得距离上最接近的伪造图像。总结一下，W_F^+指的是通过样本采样和MLP转换得到的用于生成伪造图像集X_F的一组潜码。而W_F^i指的是X_F中单个伪造图像的具体潜码，并在特征空间中与真实图像进行配对，以训练k-NN分类器并进行后续的优化和匿名化处理。潜码W_F^i经过修改后得到W_A^i，旨在生成最终的匿名化图像X_A。
在优化过程中，W_A^i会调整以确保生成的图像在特征空间中与真实图像X_R的特征表示接近，这样可以保留关键属性，同时通过最小化身份损失L_{id}来混淆身份特征。这个过程包括对比真实图像和生成图像的特征表示，然后通过梯度下降等方法优化W_A^i。在实际应用中，这种优化可能会在FaRL特征空间内进行，其中考虑了属性保持损失L_{att}和身份损失L_{id}。


## 关于数据集问题
下载CelebAMask-HQ的数据集下有CelebA-HQ-to-CelebA-mapping.txt、CelebAMask-HQ-attribute-anno.txt这两个文件，在原始的CelebA数据集下的annotations文件夹下有list_eval_partition.txt文件，组合到一起，按照原github中显示的CelebA-HQ数据集排列方式，把CelebAMask-HQ的数据集文件夹名字重命名一下。

