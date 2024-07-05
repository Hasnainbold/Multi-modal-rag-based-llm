import torch
from transformers import AutoFeatureExtractor, AutoModel
import weaviate
import weaviate.classes as wvc
from PIL import Image
import torchvision.transforms as T
from datasets import Dataset


class ImageDatabase:
  def __init__(self, extractor, model, db_url, db_key):
    self.extractor = extractor
    self.model = model
    hidden_dim = self.model.config.hidden_size
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.client = weaviate.connect_to_wcs(cluster_url=db_url, auth_credentials=weaviate.auth.AuthApiKey(db_key))
    self._model_prep()

  def _model_prep(self):
    self.transformation_chain = T.Compose(
        [
            T.Resize(int((256 / 224) * self.extractor.size["height"])),
            T.CenterCrop(self.extractor.size["height"]),
            T.ToTensor(),
            lambda x: x[:3, :, :] if x.shape[0] >= 3 else x.repeat(3, 1, 1),
            T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std)
        ]
    )

  def _get_image_embedding(self, image): # image has to be PIL object
    image_transformed = self.transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(self.device)}
    with torch.no_grad():
      embeddings = self.model(**new_batch).last_hidden_state[:, 0].cpu()
    return embeddings

  def add_image(self, image_file_path):
    image = Image.open(image_file_path)
    embeddings = self._get_image_embedding(image)
    db_obj = [wvc.data.DataObject(properties={"image_file_path":image_file_path},vector=embeddings)]
    self.client.collections.get("Images").data.insert_many(db_obj)

  def add_image_batch(self, im_dataset):  # expected to contain image_file_path and image PIL object
    def extract_embeddings(model: torch.nn.Module):
      """Utility to compute embeddings."""
      device = model.device

      def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
          [self.transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = self.model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

      return pp

    batch_size = im_dataset.num_rows//5
    extract_fn = extract_embeddings(self.model.to(self.device))
    image_embs = im_dataset.map(extract_fn, batched=True, batch_size=batch_size)

    question_objs = list()
    for d in image_embs:
      question_objs.append(wvc.data.DataObject(
        properties={
            "image_file_name":d["image_file_name"]
        },
        vector=d["embeddings"]
      ))
    self.client.collections.get("Images").data.insert_many(question_objs)

  def query(self, image, top_k=2):
    embed = self._get_image_embedding(image)

    response = self.client.collections.get("Images").query.near_vector(
      near_vector=embed.tolist()[0],
      limit=top_k,
      return_metadata=wvc.query.MetadataQuery(certainty=True)
    )
    return [o.properties['image_file_name'] for o in response.objects]


class TextDatabase:
    def __init__(self, embedder, db_url, db_key):
        self.embedder = embedder
        self.client = weaviate.connect_to_wcs(cluster_url=db_url,auth_credentials=weaviate.auth.AuthApiKey(db_key))
        self.dataset = ""

    def add_image_batch(self, im_dataset):
        def embed(batch):
          return {"embeddings":self.embedder.embed_documents(batch["image_file_name"])}

        batch_size = im_dataset.num_rows//5
        name_embeds = im_dataset.map(embed, batched=True, batch_size=batch_size)
        image_objs = [
          wvc.data.DataObject(properties={"image_file_name":d["image_file_name"]},vector=d["embeddings"])
          for d in name_embeds
        ]

        if self.dataset == "":
          self.dataset = Dataset.from_dict({"image_file_name":im_dataset["image_file_name"],"image":im_dataset["image"]})
        else:
          for d in name_embeds:
            self.dataset = self.dataset.add_item({"image_file_name":d["image_file_name"], "image":d["image"]})
        self.client.collections.get("Texts").data.insert_many(image_objs)

    def add_image(self,image_file):
        image = Image.open(image_file)
        name_embed = self.embedder.embed_query(image_file)
        db_obj = [wvc.data.DataObject(properties={"image_file_name":image_file},vector=name_embed)]

        if self.dataset == "":
            self.dataset = Dataset.from_dict({"image_file_name":[image_file],"image":[image]})
        else:
          self.dataset = self.dataset.add_item({"image_file_name":image_file, "image":image})

        self.client.collections.get("Texts").data.insert_many(db_obj)

    def query(self, text, top_k=2):
        embed = self.embedder.embed_query(text)

        response = self.client.collections.get("Texts").query.near_vector(
          near_vector=embed,
          limit=top_k,
          return_metadata=wvc.query.MetadataQuery(certainty=True)
        )
        names = [o.properties["image_file_name"] for o in response.objects]
        return [self.dataset['image'][self.dataset['image_file_name'].index(n)] for n in names if n is not None]
