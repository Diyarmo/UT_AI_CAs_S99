# -*- coding: utf-8 -*-
"""Copy of Recitation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SAQLmOTqL82HH5qbnEGjZK6PExR9osV6
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
from datetime import timedelta
from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import os
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix

from google.colab import drive
drive.mount('/content/gdrive')

!unzip -q /content/gdrive/My\ Drive/categorized_products.zip

"""##1"""

main_path, classes, _ = list(os.walk("/content/categorized_products"))[0]
plt.figure(figsize=(16, 8))
for i, cls in enumerate(np.random.choice(classes, 12)):
  class_path = os.path.join(main_path, cls)
  plt.subplot(2,6, i+1)
  img_path = os.path.join(class_path, np.random.choice(list(os.walk(class_path))[0][2]))
  plt.imshow(plt.imread(img_path))
  plt.title(cls)



"""##2"""

class ToTensor(object):
    """Convert PIL Images in sample to pytorch Tensors."""

    def __call__(self, image):
        image = np.array(image, dtype=np.float32)
        # numpy image: H x W
        return torch.from_numpy(image)

transform = transforms.Compose([transforms.Grayscale(),
                                ToTensor()
                               ]) 

dataset = ImageFolder( "/content/categorized_products"
                      , transform=transform)

"""### Spliting data"""

validation_split = 0.2
targets = [x[1] for x in dataset ]
target_names = [classes[x] for x in targets]
tnc = Counter(target_names)
indices = list(range(len(dataset))) 

train_indices, test_indices = train_test_split(indices, test_size=validation_split, shuffle=True, stratify=targets)



"""### Bar plot of count of each class in data"""

keys = np.array(list(tnc.keys()))
values = list(tnc.values())
plt.figure(figsize=(20, 16))
sns.barplot(y=keys, x=values , order=keys[np.argsort(values)])

"""### Metrics"""

def get_labels_and_predicts(model, loader):
  all_predicts = []
  all_labels = []

  # Iterate through test dataset
  with torch.no_grad():
    for images, labels in loader:
      outputs = model(images.to(device))
      _, predicted = torch.max(outputs.data, 1)
      all_predicts += (predicted.tolist())
      all_labels += (labels.tolist())
  return all_labels, all_predicts
def show_model_metrics(model, test_loader, train_loader):

  train_labels, train_predicts = get_labels_and_predicts(model, train_loader)
  test_labels, test_predicts = get_labels_and_predicts(model, test_loader)
  print("Acc on train data:", accuracy_score(y_true=train_labels, y_pred=train_predicts))
  print("ACC on test data:", accuracy_score(y_true=test_labels, y_pred=test_predicts))
  cm = confusion_matrix(y_true=test_labels, y_pred=test_predicts)
  precisions = (np.diag(cm) / cm.sum(axis=0))
  precisions = np.nan_to_num(precisions)
  plt.figure(figsize=(15, 10))
  sns.barplot(y=classes, x=precisions, order=np.array(classes)[np.argsort(precisions)])
  plt.xlabel("Precision")
  plt.ylabel("Class")
  plt.show()
  plt.figure(figsize=(8, 6))
  plt.plot(np.arange(1, num_epochs+1), model.losses)
  plt.xlabel("Epoch")
  plt.ylabel("Train Loss")
  plt.show()

"""### Fit function"""

def fit(model, train_loader, device, criterion, optimizer, num_epochs=10, verbose=True):

  total_time = 0.
  model.losses = []
  for epoch in range(num_epochs):
      train_loss = 0.
      d1 = datetime.now()
      for images, labels in train_loader:
          
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        train_loss += loss.item()

      average_loss = train_loss / len(train_loader)
      d2 = datetime.now()
      delta = d2 - d1
      seconds = float(delta.total_seconds())
      total_time += seconds
      if verbose:
        print('epoch %d, train_loss: %.3f, time elapsed: %s seconds' % (epoch + 1, average_loss, seconds))
      model.losses.append(average_loss)
  print('total training time: %.3f minutes' % (total_time / 60))



batch_size = 64

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16)

"""##3

###Designing the Model
"""

class Model(nn.Module):
    def __init__(self, class_num, act=F.relu):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(1 * 80 * 60, 3000)
        self.act1 = act

        # self.layer2 = nn.Linear(10000, 8000)
        # self.act2 = act

        # self.layer3 = nn.Linear(8000, 5000)
        # self.act3 = act

        # self.layer4 = nn.Linear(5000, 2000)
        # self.act4 = act

        self.layer5 = nn.Linear(3000, 1500)
        self.act5 = act

        self.layer6 = nn.Linear(1500, 750)
        self.act6 = act

        self.layer7 = nn.Linear(750, 300)
        self.act7 = act

        self.layer8 = nn.Linear(300, class_num)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.act1(x)

        # x = self.layer2(x)
        # x = self.act2(x)

        # x = self.layer3(x)
        # x = self.act3(x)

        # x = self.layer4(x)
        # x = self.act4(x)

        x = self.layer5(x)
        x = self.act5(x)

        x = self.layer6(x)
        x = self.act6(x)

        x = self.layer7(x)
        x = self.act7(x)

        x = self.layer8(x)

        return x

"""### A more complex model"""

# class Model(nn.Module):
#     def __init__(self, class_num, act=F.relu):
#         super(Model, self).__init__()

#         self.layer1 = nn.Linear(1 * 80 * 60, 8000)
#         self.act1 = act

#         # self.layer2 = nn.Linear(10000, 8000)
#         # self.act2 = act

#         self.layer3 = nn.Linear(8000, 5000)
#         self.act3 = act

#         self.layer4 = nn.Linear(5000, 2000)
#         self.act4 = act

#         self.layer5 = nn.Linear(2000, 1500)
#         self.act5 = act

#         self.layer6 = nn.Linear(1500, 750)
#         self.act6 = act

#         self.layer7 = nn.Linear(750, 300)
#         self.act7 = act

#         self.layer8 = nn.Linear(300, class_num)

#     def forward(self, x):

#         x = x.view(x.size(0), -1)

#         x = self.layer1(x)
#         x = self.act1(x)

#         # x = self.layer2(x)
#         # x = self.act2(x)

#         x = self.layer3(x)
#         x = self.act3(x)

#         x = self.layer4(x)
#         x = self.act4(x)

#         x = self.layer5(x)
#         x = self.act5(x)

#         x = self.layer6(x)
#         x = self.act6(x)

#         x = self.layer7(x)
#         x = self.act7(x)

#         x = self.layer8(x)

#         return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""###3-A"""

model = Model(len(classes))
model = model.to(device)

sum_params = 0
params = list(model.parameters())
for i in range(0, len(params), 2):
  num_params = params[i].size()[0] * params[i].size()[1] + params[i+1].size()[0]
  sum_params += num_params
  print("Layer {}: {} <= {} * {} + {} ".format(i//2 + 1, num_params, params[i].size()[0], params[i].size()[1], params[i+1].size()[0]))
print("Total : ", sum_params)

"""<html>
همانگونه که نشان داده شده تعداد پارامتر‌های هر لایه مساوی است با حاصلضرب تعداد نورون‌های آن لایه در تعداد نورون‌های لایه‌ی بعدی به علاوه‌ی تعداد بایاس‌ها که برابر تعداد نورون‌های آن لایه است.
</html>
"""



"""### 3-B"""

learning_rate = 0.01
num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
fit(model, train_loader, device, criterion, optimizer, num_epochs)

show_model_metrics(model, test_loader, train_loader)

"""<html>
 <p dir="rtl">
به دلیل اینکه گرادیان ها با ورودی ارتباط مستقیم دارند(در relu
دقیقا برابر با ورودی است) پس بزرگ بودن ورودی‌ها و نرمال نبودن مقادیر آن میتواند موجب تغییرات بسیار بزرگ در وزن‌ها شود و باعث می‌شود پس از مدتی پدیده‌ی 
exploding weights 
رخ بدهد.
در این حالت وزن‌ها آنقد بزرگ می‌شوند 
که مقدار
loss
برابر
nan 
خواهد شد و دیگر آپدیت هم ممکن نیست.

</p>
</html>

##4
"""

class ToTensorAndNormalize(object):
    """Convert PIL Images in sample to pytorch Tensors."""
    def __call__(self, image):
        image = np.array(image, dtype=np.float32) / 255
        # numpy image: H x W
        return torch.from_numpy(image)

new_transform = transforms.Compose([transforms.Grayscale(),
                                    ToTensorAndNormalize(),
                                    ]) 

normal_dataset = ImageFolder( "/content/categorized_products"
                      , transform=new_transform)

validation_split = 0.2
targets = [x[1] for x in normal_dataset ]
target_names = [classes[x] for x in targets]
tnc = Counter(target_names)
indices = list(range(len(normal_dataset))) 

train_indices, test_indices = train_test_split(indices, test_size=validation_split, shuffle=True, stratify=targets)

batch_size = 64

train_normal_sampler = SubsetRandomSampler(train_indices)
test_normal_sampler = SubsetRandomSampler(test_indices)

train_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=train_normal_sampler, num_workers=16)
test_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=test_normal_sampler, num_workers=16)

learning_rate = 0.01
num_epochs = 10
model = Model(len(classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs)



show_model_metrics(model, test_normal_loader, train_normal_loader)

"""در این قسمت به دلیل نرمالایز شدن  دیتا مشکل بالا رفع شده است.

##5

###5 - A
"""

learning_rate = 0.01
num_epochs = 10
model = Model(len(classes))
model = model.to(device)
def func(x):
  if type(x) == torch.nn.modules.linear.Linear:
    x.weight.data.fill_(0)
    x.bias.data.fill_(0)
model = model.apply(func)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs)

show_model_metrics(model, test_normal_loader, train_normal_loader)

tnc.most_common()[0]

"""<html>
 <p dir="rtl">
وقتی که وزن‌ها صفر باشند همواره خروجی نورون‌ها صفر خواهد بود لذا هیچ آپدیتی رخ نخواهد داد و تنها بایاس لایه‌ی آخر است که تاثیر گذار می‌شود و مدل تبدیل به یک طبقه‌بند خطی می‌شوذ که بهترین کاری که میتواند بکند طبقه‌بندی تمام داده ها به عنوان بزرگترین کلاس است (Flip Flops).
</p>
</html>

### 5 - B

یکی از بهترین راه‌حل ها استفاده از وزن‌های نرمال با میانگین صفر و مقادیر نزدیک صفر است که هم مدل آموزش داده شود و هم اینکه دچار انفجار وزن‌ها نشویم و همچنین شروع از نقاط رندوم باشد که احتمال گیر کردن در نقطه‌ی بهینه‌ی محلی در اجراهای مختلف کمتر شود.

اینکه توزیع رندوم وزن‌ها چگونه باشد نیز متفاوت است.
میتوان یونیفرم با میانگین صفر انتخاب کرد.
توزیع گوسی نیز میتواند مفید باشد.
تعیین پارامترهای این توزیع‌ها بهتر است به نحوی صورت گیرد که حاصل‌جمع ورودی‌های هر نورون بین -4 تا 4 باشد.

##6

###6 - A
"""

learning_rates = [0.0001, 0.0003,  0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3]
for lr in learning_rates:
  num_epochs = 10
  model = Model(len(classes))
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)
  train_labels, train_predicts = get_labels_and_predicts(model, train_normal_loader)
  test_labels, test_predicts = get_labels_and_predicts(model, test_normal_loader)
  print("Learning rate:", lr)
  print("Acc on train data:", accuracy_score(y_true=train_labels, y_pred=train_predicts))
  print("ACC on test data:", accuracy_score(y_true=test_labels, y_pred=test_predicts))

"""<html>
 <p dir="rtl">
بهترین مقدار برای نرخ یادگیری 0.03 است.
</p>
</html>
"""

learning_rate = 0.03
num_epochs = 10
model = Model(len(classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)

show_model_metrics(model, test_normal_loader, train_normal_loader)

"""### 6 - B

<html>
<p dir="rtl">

وقتی نرخ یادگیری بسیار پایین باشد مدل قدم‌های بسیار کوچکتری جهت رسیدن به نقطه‌ی بهینه بر می‌دارد در نتیجه به دیتاهای بیشتر یا تعداد دورهای بیشتری برای یادگیری نیاز است تا به نقطه‌ی بهینه برسد.
در صورتی که این نرخ بالا باشد گام های مدل بسیار بزرگ خواهد بود و مدل یا حول نقطه‌ی بهینه جهش دارد و به آن نمی‌رسد یا به نقطه‌ی بهینه‌ی محلی همکرا میشود.
با نرخ پایین بالا بردن تعداد دورها می‌توان به نتیجه‌ی مناسب رسید ولی با نرخ بالا این احتمال وحود دارد که هیج‌وقت همگرا نشود.
</p>
</html>

## 7

### 7 - A
"""

for batch_size in [32, 128]:
  train_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=train_normal_sampler, num_workers=16)
  test_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=test_normal_sampler, num_workers=16)
  learning_rate = 0.03
  num_epochs = 10
  model = Model(len(classes))
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)
  train_labels, train_predicts = get_labels_and_predicts(model, train_normal_loader)
  test_labels, test_predicts = get_labels_and_predicts(model, test_normal_loader)
  print("Batch size rate:", batch_size)
  print("Acc on train data:", accuracy_score(y_true=train_labels, y_pred=train_predicts))
  print("ACC on test data:", accuracy_score(y_true=test_labels, y_pred=test_predicts))

"""<html>
<p dir="rtl">
مقدار بهینه‌ برابر ۳۲ است.

چون در هر بچ گرادیان‌ها جمع می‌شوند با بچ سایز بزرگتر دقت این آپدیت‌ها کمتر می‌شود ولی چون دفعات کمتری آپدیت انجام می‌شود زمان کمتری نیاز دارد(با بج سایز ‍۱۲۸ به ۲.۱ دقیقه زمان نیاز بود،
ولی با بچ سایز ۳۲ یادگیری در ۲.۷ دقیقه انجام شده است.

اگر بچ سایز بسیار بزرگ باشد چون گرادیان برای همه‌ی داده‌ها حساب شده و جمع 
می‌شود یادگیری بسیار کند می‌شود و ولی هر ایپاک سریع تمام می‌شود
اگر بچ سایز بسیار کم باشد مثلا ۱، بعد از دیدن هر دیتا مدل آپدیت می‌شود که باعث می‌شود یادگیری دقیق‌تر و در تعداد دور کمتری انجام شود ولی از نظر زمانی بسیار طول خواهد کشید.
</p>
</html>
"""

batch_size = 32
train_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=train_normal_sampler, num_workers=16)
test_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=test_normal_sampler, num_workers=16)
learning_rate = 0.03
num_epochs = 10
model = Model(len(classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)

show_model_metrics(model, test_normal_loader, train_normal_loader)

"""### 7 - B"""

batch_size = 128
train_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=train_normal_sampler, num_workers=16)
test_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=test_normal_sampler, num_workers=16)
  
for learning_rate in [0.01, 0.025, 0.035, 0.05]:
  num_epochs = 10
  model = Model(len(classes))
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)
  train_labels, train_predicts = get_labels_and_predicts(model, train_normal_loader)
  test_labels, test_predicts = get_labels_and_predicts(model, test_normal_loader)
  print("Learning rate rate:", learning_rate)
  print("Acc on train data:", accuracy_score(y_true=train_labels, y_pred=train_predicts))
  print("Acc on test data:", accuracy_score(y_true=test_labels, y_pred=test_predicts))

"""<html>
<p dir="rtl">
گفته می‌شود که بهتر است با افزایش batch_size 
نرخ یادگیری نیز افزایش یابد همانطور 
که در بالا می‌بینید
با افزایش نرخ یادگیری عملکرد نیز بهتر می‌شود.
</p>
</html>

## 8

### 8 - A


<html>
<p dir="rtl">
در این روش به جای استفاده از گرادیان آخرین گام از گرادیان‌های قبلی نیز استفاده میکنیم و به این صورت گام‌های مطمئن تری برداشته می‌شود.
از طرف دیگر این میتواند موجب شود که گام‌های ما دقیقا در راستای نقطه‌ی بهینه نباشند و احتمال خطا مقداری بالاتر می‌رود.

</p>
</html>
"""

batch_size = 64
train_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=train_normal_sampler, num_workers=16)
test_normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=batch_size, sampler=test_normal_sampler, num_workers=16)
  
for momentum in [0.5, 0.9, 0.98]:
  num_epochs = 10
  learning_rate = 0.03
  model = Model(len(classes))
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
  fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)
  train_labels, train_predicts = get_labels_and_predicts(model, train_normal_loader)
  test_labels, test_predicts = get_labels_and_predicts(model, test_normal_loader)
  print("momentum:", momentum)
  print("Acc on train data:", accuracy_score(y_true=train_labels, y_pred=train_predicts))
  print("Acc on test data:", accuracy_score(y_true=test_labels, y_pred=test_predicts))

"""همانطور که میبیند افزایش ممان لزوما باعث بهبود عملکرد نمی‌شود و اگر از مقادیری بیشتر شود عملا مدل هر بار تمام گرادیان‌های قبلی را با هم جمع میکند و در آن راستا حرکت میکند که حرکت دقیقی به سمت نقطه‌ی بهینه نیست از طرف دیگر مقدار 0.5 به نظر مناسب میرسد و حتی عمکلرد مدل را نیز بهبود بخشیده است چون به مقداری از گرادیان‌های قبلی استفاده میکند که گام‌های مطمئن‌تری بردارد و آنقدری تاثیرشان بالا نیست که باعث گم شدن مدل در فضای وزن‌ها شود."""



"""##9"""

num_epochs = 20
learning_rate = 0.03
momentum = 0.5
model = Model(len(classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=True)

show_model_metrics(model, test_normal_loader, train_normal_loader)

"""<html>
<p dir="rtl">
تعداد دور‌های بیشتر منجر به همگرایی بیشتر مدل به نقطه‌ی بهینه می‌شود، زیرا در هر دور دوباره با داده‌ها آموزش داده می‌شود و دقیق‌تر شود به عبارت دیگر اصولا 
SGD
روشی 
iterative
است و تضمینی ندارد که در اولین دور به نقطه‌ی بهینه‌ برسد. پس از مدتی دیگر به دلیل ناتوانی مدل در بهبود بیشتر(عدم ظرفیت بیشتر مدل در یادگیری) و یا اینکه دیگر در دیتا‌ها اطلاعات بیشتری موجود نیست مدل سرعت بهبودش کاهش یافته و پس مدتی دیگر بهبودی مشاهده نمی‌شود، در دور ۱۹ و ۲۰ مشاهده می‌شود که نمودار میزان لاس دارد افقی ‌می‌شود و تقریبا دیگر جای پیشرفتی وجود ندارد. 

</p>
</html>

## 10
"""

num_epochs = 10
learning_rate = 0.03
momentum = 0.5
model = Model(len(classes), F.tanh)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)

show_model_metrics(model, test_normal_loader, train_normal_loader)

num_epochs = 10
learning_rate = 0.03
momentum = 0.5
model = Model(len(classes), F.leaky_relu)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=False)

show_model_metrics(model, test_normal_loader, train_normal_loader)

"""<html>
<p dir="rtl">
در هر دو حالت تنها مقداری دقت پایین آمده است.
دلیل آن نیز میتواند در شباهت تقریبی هر سه‌ی این مدل‌ها در ظاهر نمودارشان باشد.
</p>
</html>

##11

### 11 - A

<html>
<p dir="rtl">
 در واقع regularizer با گذاشتن محدودیت روی مقدار وزن‌ها 
سعی دارد ضرایب را به سمت صفر بکشاند تا مدل را به سمت سادگی بیشتر(با حفظ عملکرد قبلی) بکشاند.
و در واقع مدل در نقطه میانه‌ای بین پیچیدگی و عملکرد مناسب قرار می‌گیرد و دیگر مدل بسیار پیچیده(محتمل اورفیت) با عملکرد بسیار بالا ایجاد نمی‌شود.
</p>
</html>

### 11 - B
Weight decay update rule is

$W = (1-\lambda)W - \alpha\Delta C_0$

and L2 Regularization is

$C = C_0 + \frac{\lambda}{2}||W||_2^2$

we then calculate the gradient of loss function

$\Delta C = \Delta C_0 + \lambda W$

then in gradient desenct weights update we have

$W = W - \alpha\Delta C$

$W = W - \alpha(\Delta C_0 + \lambda W)$

$W = W - \alpha\Delta C_0 - \alpha\lambda W$

$W = (1 - \alpha\lambda)W - \alpha\Delta C_0$

which is pretty similar to weight decay update rule. let the lambda in weight decay updte rule be $\lambda' = \alpha \lambda$
"""

num_epochs = 10
learning_rate = 0.03
momentum = 0.5
weight_decay = 0.1
model = Model(len(classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay )
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=True)

show_model_metrics(model, test_normal_loader, train_normal_loader)

"""<html>
<p dir="rtl">
weight decay
برابر 0.1 بسیار زیاد است، و به این معنی است که در هر آپدیت فقط 0.9 از وزن قبلی باقی بماند و در تعداد کمی آپدیت (حدود ۳۵ آپدیت) میتواند وزن برابر ۱ را به صفر تبدیل کند که با توجه به تعداد epochها
و تعداد دفعات آپدیت در هر دور بسیار محتمل است که عملکرد مدل به شدت مختل شود.
همان‌گونه که مشاهده میکنید تقریبا وزن‌ها صفر شده‌اند و مدل مانند سوال ۵ به یک طبقه‌بند خطی تبدیل شده است.
</p>
</html>
"""

num_epochs = 10
learning_rate = 0.03
momentum = 0.5
weight_decay = 0.01
model = Model(len(classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay )
fit(model, train_normal_loader, device, criterion, optimizer, num_epochs, verbose=True)

show_model_metrics(model, test_normal_loader, train_normal_loader)

"""<html>
<p dir="rtl">
weight decay
برابر 0.01 از حالت قبل بسیار کمتر است و عملکرد مدل مانند حالت قبل مختل نمی‌شود ولی باید این نکته را در نظر داشت که مدل ما در ۱۰
epoch
اورفیت نشده است که این روش به ما کمکی بکند لذا تنها مقداری از دقت مدل نسبت به سوال ۸ می‌کاهد.

شاید اگر مدلمان را به تعداد دور زیادی آموزش دهیم که کم کم دقت روی ترین و دقت روی تست شروع به فاصله گرفتن کنند و اورفیت رخ بدهد استفاده از این روش سبب افزایش دقت روی ترین شود.
</p>
</html>
"""

model



