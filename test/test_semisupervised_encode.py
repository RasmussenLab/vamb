import unittest
import numpy as np
import tempfile
import vamb


class TestDataLoader(unittest.TestCase):
    def test_permute_indices(self):
        indices = vamb.semisupervised_encode.permute_indices(10, 25, seed=1)
        set_10 = set(range(10))
        self.assertTrue(len(indices) == 25)
        self.assertTrue(set(indices[:10]) == set_10)
        self.assertTrue(set(indices[10:20]) == set_10)
        self.assertTrue(set(indices[20:]).issubset(set_10))


class TestVAEVAE(unittest.TestCase):
    N_contigs = 111
    tnfs = np.random.random((N_contigs, 103)).astype(np.float32)
    rpkms = np.random.random((N_contigs, 14)).astype(np.float32)
    domain = "d_Bacteria"
    phyla = ["f_1", "f_2", "f_3"]
    classes = {
        "f_1": ["c_11", "c_21", "c_31"],
        "f_2": ["c_12", "c_22", "c_32"],
        "f_3": ["c_13", "c_23", "c_33"],
    }
    lengths = np.random.randint(2000, 5000, size=N_contigs)

    def make_random_annotation(self):
        phylum = np.random.choice(self.phyla, 1)[0]
        clas = np.random.choice(self.classes[phylum], 1)[0]
        if np.random.random() <= 0.2:
            return vamb.taxonomy.ContigTaxonomy.from_semicolon_sep(
                ";".join([self.domain])
            )
        if 0.2 < np.random.random() <= 0.5:
            return vamb.taxonomy.ContigTaxonomy.from_semicolon_sep(
                ";".join([self.domain, phylum])
            )
        return vamb.taxonomy.ContigTaxonomy.from_semicolon_sep(
            ";".join([self.domain, phylum, clas])
        )

    def make_random_annotations(self):
        return [self.make_random_annotation() for _ in range(self.N_contigs)]

    def test_make_graph(self):
        annotations = self.make_random_annotations()
        nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(annotations)
        print(nodes, ind_nodes, table_parent)
        self.assertTrue(
            set(nodes).issubset(
                set(
                    [
                        "root",
                        "d_Bacteria",
                        "f_1",
                        "f_2",
                        "f_3",
                        "c_11",
                        "c_21",
                        "c_31",
                        "c_12",
                        "c_22",
                        "c_32",
                        "c_13",
                        "c_23",
                        "c_33",
                    ]
                )
            )
        )
        for p, cls in self.classes.items():
            for c in cls:
                for f in self.phyla:
                    # Since the taxonomy is generated randomly, we can't guarantee
                    # that each run will have all the clades.
                    if any(i not in ind_nodes for i in (p, c, f)):
                        continue
                    self.assertTrue(ind_nodes.get(f, -666) < ind_nodes.get(c, 666))
                    self.assertTrue(table_parent[ind_nodes[f]] == 1)
                    self.assertTrue(table_parent[ind_nodes[c]] == ind_nodes[p])

    def test_encoding(self):
        nlatent = 10
        batchsize = 10
        nepochs = 2
        annotations = self.make_random_annotations()
        nodes, ind_nodes, table_parent = vamb.taxvamb_encode.make_graph(annotations)

        classes_order = np.array([a.ranks[-1] for a in annotations])
        targets = np.array([ind_nodes[i] for i in classes_order])

        vae = vamb.taxvamb_encode.VAEVAEHLoss(
            self.rpkms.shape[1],
            len(nodes),
            nodes,
            table_parent,
            nlatent=nlatent,
            cuda=False,
        )

        dataloader_vamb = vamb.encode.make_dataloader(
            self.rpkms,
            self.tnfs,
            self.lengths,
            batchsize=batchsize,
            cuda=False,
        )
        dataloader_joint = vamb.taxvamb_encode.make_dataloader_concat_hloss(
            self.rpkms,
            self.tnfs,
            self.lengths,
            targets,
            len(nodes),
            table_parent,
            batchsize=batchsize,
            cuda=False,
        )
        dataloader_labels = vamb.taxvamb_encode.make_dataloader_labels_hloss(
            self.rpkms,
            self.tnfs,
            self.lengths,
            targets,
            len(nodes),
            table_parent,
            batchsize=batchsize,
            cuda=False,
        )

        shapes = (self.rpkms.shape[1], 103, 1, len(nodes))
        dataloader = vamb.taxvamb_encode.make_dataloader_semisupervised_hloss(
            dataloader_joint,
            dataloader_vamb,
            dataloader_labels,
            len(nodes),
            table_parent,
            shapes,
            666,
            batchsize=batchsize,
            cuda=False,
        )
        with tempfile.TemporaryFile() as modelfile:
            vae.trainmodel(
                dataloader,
                nepochs=nepochs,
                modelfile=modelfile,
                batchsteps=[],
            )

        latent_both = vae.VAEJoint.encode(dataloader_joint)
        self.assertEqual(latent_both.dtype, np.float32)
        self.assertEqual(latent_both.shape, (len(self.rpkms), nlatent))
