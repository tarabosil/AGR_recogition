from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics._scorer import average
from sklearn.preprocessing import LabelBinarizer


class Evaluator:

    def roc_curve(self, y_true, preds, type):
        """ Plot ROC curve """
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_keras = auc(fpr, tpr)
        print(auc_keras)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(f'images/{type}_roc.png')
        plt.show()

    def multi_roc_curve(self, y_test, preds_multi, type, num_classes, labels):
        """ ROC curve and AUC value for multi class prediction """
        fpr = dict()
        tpr = dict()
        auc_roc = dict()
        colors = ['lightskyblue', 'orchid', 'royalblue', 'mediumspringgreen', 'hotpink', 'limegreen']

        for i in range(num_classes[type]):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds_multi[:, i])
            auc_roc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(num_classes[type]):
            plt.plot(fpr[i], tpr[i], color=colors[i], label=f'{labels[i]} (AUC = {round(auc_roc[i], 3)})')
        plt.title('ROC curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(f'images/{type}_roc.png')
        plt.show()

    def plot_gender_barchart(self, genders):
        plt.figure()

        plt.bar([0,1], [genders.count('0'), genders.count('1')], color=['lightskyblue', 'orchid'])
        plt.xticks([0,1], ['Male', 'Women'])
        plt.title('Gender distribution')
        plt.show()

    def plot_race_barchart(self, races):

        plt.figure()
        colors = ['lightskyblue', 'orchid', 'royalblue', 'mediumspringgreen', 'hotpink']

        plt.bar([0, 1, 2, 3, 4], [races.count('0'), races.count('1'), races.count('2'), races.count('3'), races.count('4')], color=colors)
        plt.xticks([0, 1, 2, 3, 4], ['White', 'Black', 'Asian', 'Indian', 'Others'])
        plt.title('Race distribution')
        plt.show()

    def plot_age_barchart(self, ages):

        plt.figure()
        colors = ['lightskyblue', 'orchid', 'royalblue', 'mediumspringgreen', 'hotpink', 'limegreen']

        plt.bar([0, 1, 2, 3, 4, 5], [ages.count(0), ages.count(1), ages.count(2), ages.count(3), ages.count(4), ages.count(5)], color=colors)
        plt.xticks([0, 1, 2, 3, 4, 5], ['Children', 'Youth', 'Adults', 'Middle age', 'Old', 'Very old'])
        plt.title('Age distribution')
        plt.show()

