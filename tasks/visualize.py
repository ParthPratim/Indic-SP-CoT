import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from src.utils import fix_seed
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument(
        "--dataset", type=str, default="hotpot-qa",
        choices=["hotpot-qa", "strategy-qa", "cweb-qa", "fool-me-twice", "hover", "feverous"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument(
        "--num_clusters", type=str, default="[2,4,6,8]", help="number of clusters"
    )
    parser.add_argument(
        "--pred_file", type=str, default="log/multiarith_zero_shot_cot.log",
        help="use the reasoning chains generated by zero-shot-cot."
    )
    parser.add_argument(
        "--demo_save_dir", type=str, default="demos", help="where to save the contructed demonstrations"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to use"
    )

    args = parser.parse_args()
    args.demo_save_dir = os.path.join(args.demo_save_dir, args.dataset)
    os.makedirs(args.demo_save_dir, exist_ok=True)

    return args


def main(args):
    fix_seed(args.random_seed)
    encoder = SentenceTransformer(args.encoder)

    corpus = []
    cots = []
    gold_answers = []
    pred_answers = []

    with open(args.pred_file) as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["prompt"])
            cots.append(data["cot"])
            gold_answers.append(data["gold_ans"])
            pred_answers.append(data["pred_ans"])

    # encode the corpus
    corpus_embeddings = encoder.encode(corpus, show_progress_bar=True, device=args.device)

    num_clusters_list = eval(args.num_clusters)

    for num_clusters in num_clusters_list:
        # cluster the corpus
        clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        clustered_idx = [[] for i in range(num_clusters)]
        clustered_dists = [[] for i in range(num_clusters)]
        dist = clustering_model.transform(corpus_embeddings)

        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])
            clustered_idx[cluster_id].append(sentence_id)
            clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])

        demos = {}
        for i in range(num_clusters):
            demos[i] = []
            print("Cluster ", i + 1)
            tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
            top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
            if args.sampling != "center":
                random.shuffle(top_min_dist)

            for element in top_min_dist:
                min_idx = element[0]
                cot = cots[clustered_idx[i][min_idx]]
                pred_ans = pred_answers[clustered_idx[i][min_idx]]
                cot = cot.replace("\n\n", "\n").replace("\n", " ").strip()
                question = corpus[clustered_idx[i][min_idx]]
                gold_ans = gold_answers[clustered_idx[i][min_idx]]

                demo_element = {
                    "question": question,
                    "cot": cot,
                    "gold_ans": gold_ans,
                    "pred_ans": pred_ans
                }

                demos[i].append(demo_element)

        with open(os.path.join(args.demo_save_dir, args.dataset + "_nc_" + str(num_clusters) + ".json"), "w") as f:
            json.dump(demos, f, indent=4, ensure_ascii=False)

        y_km = clustering_model.fit_predict(corpus_embeddings)
        pca_model = PCA(n_components=2, random_state=args.random_seed)
        transformed = pca_model.fit_transform(corpus_embeddings)
        centers = pca_model.transform(clustering_model.cluster_centers_)

        # plt.figure(figsize=(10, 10))
        plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
        plt.scatter(centers[:, 0], centers[:, 1],
                    s=250, marker='*', label='centroids',
                    edgecolor='black',
                    c=np.arange(0, num_clusters), cmap=plt.cm.Paired, )
        plt.xticks([])
        plt.yticks([])

        plt.savefig(os.path.join(args.demo_save_dir, args.dataset + "_nc_" + str(num_clusters) + ".png"), dpi=600)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)