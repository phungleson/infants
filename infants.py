DEATH_COLUMNS = [
  "revision",
  "laterec",
  "idnumber",
  "dob_yy",
  "dob_mm",
  "dob_wk",
  "ostate",
  "ocntyfips",
  "ocntypop",
  "bfacil",
  "ubfacil",
  "bfacil3",
  "mage_impflg",
  "mage_repflg",
  "mager41",
  "mager14",
  "mager9",
  "mbcntry",
  "mrterr",
  "mrcntyfips",
  "rcnty_pop",
  "rectype",
  "restatus",
  "mbrace",
  "mrace",
  "mracerec",
  "mraceimp",
  "umhisp",
  "mracehisp",
  "mar",
  "mar_imp",
  "meduc",
  "umeduc",
  "meduc_rec",
  "fagerpt_flg",
  "fagecomb",
  "ufagecomb",
  "fagerec11",
  "fbrace",
  "fracerec",
  "ufhisp",
  "fracehisp",
  "frace",
  "lbo",
  "tbo",
  "dllb_mm",
  "dllb_yy",
  "precare",
  "precare_rec",
  "mpcb",
  "mpcb_rec6",
  "mpcb_rec5",
  "uprevis",
  "previs_rec",
  "wtgain",
  "wtgain_rec",
  "dfpc_imp",
  "cig_1",
  "cig_2",
  "cig_3",
  "tobuse",
  "cigs",
  "cig_rec6",
  "cig_rec",
  "rf_diab",
  "rf_gest",
  "rf_phyp",
  "rf_ghyp",
  "rf_eclam",
  "rf_ppterm",
  "rf_ppoutc",
  "rf_cesar",
  "rf_ncesar",
  "urf_diab",
  "urf_chyper",
  "urf_phyper",
  "urf_eclam",
  "op_cerv",
  "op_tocol",
  "op_ecvs",
  "op_ecvf",
  "uop_induc",
  "uop_tocol",
  "on_ruptr",
  "on_abrup",
  "on_prolg",
  "ld_induct",
  "ld_augment",
  "ld_nvrtx",
  "ld_steroids",
  "ld_antibio",
  "ld_chorio",
  "ld_mecon",
  "ld_fintol",
  "ld_anesth",
  "uld_meco",
  "uld_precip",
  "uld_breech",
  "md_attfor",
  "md_attvac",
  "md_present",
  "md_route",
  "md_trial",
  "ume_vag",
  "ume_vbac",
  "ume_primc",
  "ume_repec",
  "ume_forcp",
  "ume_vac",
  "rdmeth_rec",
  "udmeth_rec",
  "dmeth_rec",
  "attend",
  "apgar5",
  "apgar5r",
  "dplural",
  "imp_plur",
  "sex",
  "imp_sex",
  "dlmp_mm",
  "dlmp_dd",
  "dlmp_yy",
  "estgest",
  "combgest",
  "gestrec10",
  "gestrec3",
  "obgest_flg",
  "gest_imp",
  "dbwt",
  "bwtr14",
  "bwtr4",
  "bwtimp",
  "ab_vent",
  "ab_vent6",
  "ab_nicu",
  "ab_surfac",
  "ab_antibio",
  "ab_seiz",
  "ab_inj",
  "ca_anen",
  "ca_menin",
  "ca_heart",
  "ca_hernia",
  "ca_ompha",
  "ca_gastro",
  "ca_limb",
  "ca_cleftlp",
  "ca_cleft",
  "ca_downs",
  "ca_chrom",
  "ca_hypos",
  "uca_anen",
  "uca_spina",
  "uca_ompha",
  "uca_cleftlp",
  "uca_downs",
  "f_morigin",
  "f_forigin",
  "f_meduc",
  "f_clinest",
  "f_apgar5",
  "f_tobaco",
  "f_rf_pdiab",
  "f_rf_gdiab",
  "f_rf_phyper",
  "f_rf_ghyper",
  "f_rf_eclamp",
  "f_rf_ppb",
  "f_rf_ppo",
  "f_rf_cesar",
  "f_rf_ncesar",
  "f_ob_cervic",
  "f_ob_toco",
  "f_ob_succ",
  "f_ob_fail",
  "f_ol_rupture",
  "f_ol_precip",
  "f_ol_prolong",
  "f_ld_induct",
  "f_ld_augment",
  "f_ld_nvrtx",
  "f_ld_steroids",
  "f_ld_antibio",
  "f_ld_chorio",
  "f_ld_mecon",
  "f_ld_fintol",
  "f_ld_anesth",
  "f_md_attfor",
  "f_md_attvac",
  "f_md_present",
  "f_md_route",
  "f_md_trial",
  "f_ab_vent",
  "f_ab_vent6",
  "f_ab_nicu",
  "f_ab_surfac",
  "f_ab_antibio",
  "f_ab_seiz",
  "f_ab_inj",
  "f_ca_anen",
  "f_ca_menin",
  "f_ca_heart",
  "f_ca_hernia",
  "f_ca_ompha",
  "f_ca_gastro",
  "f_ca_limb",
  "f_ca_cleftlp",
  "f_ca_cleft",
  "f_ca_downs",
  "f_ca_chrom",
  "f_ca_hypos",
  "f_med",
  "f_wtgain",
  "f_tobac",
  "f_mpcb",
  "f_mpcb_u",
  "f_urf_diabetes",
  "f_urf_chyper",
  "f_urf_phyper",
  "f_urf_eclamp",
  "f_uob_induct",
  "f_uob_tocol",
  "f_uld_meconium",
  "f_uld_precip",
  "f_uld_breech",
  "f_u_vaginal",
  "f_u_vbac",
  "f_u_primac",
  "f_u_repeac",
  "f_u_forcep",
  "f_u_vacuum",
  "f_uca_anen",
  "f_uca_spina",
  "f_uca_omphalo",
  "f_uca_cleftlp",
  "f_uca_downs",
  "matchs",
  "aged",
  "ager5",
  "ager22",
  "manner",
  "dispo",
  "autopsy",
  "place",
  "ucod",
  "ucod130",
  "recwt",
  "eanum",
  "econdp_1",
  "econds_1",
  "enicon_1",
  "econdp_2",
  "econds_2",
  "enicon_2",
  "econdp_3",
  "econds_3",
  "enicon_3",
  "econdp_4",
  "econds_4",
  "enicon_4",
  "econdp_5",
  "econds_5",
  "enicon_5",
  "econdp_6",
  "econds_6",
  "enicon_6",
  "econdp_7",
  "econds_7",
  "enicon_7",
  "econdp_8",
  "econds_8",
  "enicon_8",
  "econdp_9",
  "econds_9",
  "enicon_9",
  "econdp_10",
  "econds_10",
  "enicon_10",
  "econdp_11",
  "econds_11",
  "enicon_11",
  "econdp_12",
  "econds_12",
  "enicon_12",
  "econdp_13",
  "econds_13",
  "enicon_13",
  "econdp_14",
  "econds_14",
  "enicon_14",
  "econdp_15",
  "econds_15",
  "enicon_15",
  "econdp_16",
  "econds_16",
  "enicon_16",
  "econdp_17",
  "econds_17",
  "enicon_17",
  "econdp_18",
  "econds_18",
  "enicon_18",
  "econdp_19",
  "econds_19",
  "enicon_19",
  "econdp_20",
  "econds_20",
  "enicon_20",
  "ranum",
  "record_1",
  "record_2",
  "record_3",
  "record_4",
  "record_5",
  "record_6",
  "record_7",
  "record_8",
  "record_9",
  "record_10",
  "record_11",
  "record_12",
  "record_13",
  "record_14",
  "record_15",
  "record_16",
  "record_17",
  "record_18",
  "record_19",
  "record_20",
  "d_restatus",
  "stoccfipd",
  "cntocfipd",
  "stresfipd",
  "drcnty",
  "cntyrfpd",
  "cntrsppd",
  "hospd",
  "weekdayd",
  "dthyr",
  "dthmon",
]

BIRTH_COLUMNS = [
  "revision",
  "laterec",
  "idnumber",
  "dob_yy",
  "dob_mm",
  "dob_wk",
  "ostate",
  "ocntyfips",
  "ocntypop",
  "bfacil",
  "ubfacil",
  "bfacil3",
  "mage_impflg",
  "mage_repflg",
  "mager41",
  "mager14",
  "mager9",
  "mbcntry",
  "mrterr",
  "mrcntyfips",
  "rcnty_pop",
  "rectype",
  "restatus",
  "mbrace",
  "mrace",
  "mracerec",
  "mraceimp",
  "umhisp",
  "mracehisp",
  "mar",
  "mar_imp",
  "meduc",
  "umeduc",
  "meduc_rec",
  "fagerpt_flg",
  "fagecomb",
  "ufagecomb",
  "fagerec11",
  "fbrace",
  "fracerec",
  "ufhisp",
  "fracehisp",
  "frace",
  "lbo",
  "tbo",
  "dllb_mm",
  "dllb_yy",
  "precare",
  "precare_rec",
  "mpcb",
  "mpcb_rec6",
  "mpcb_rec5",
  "uprevis",
  "previs_rec",
  "wtgain",
  "wtgain_rec",
  "dfpc_imp",
  "cig_1",
  "cig_2",
  "cig_3",
  "tobuse",
  "cigs",
  "cig_rec6",
  "cig_rec",
  "rf_diab",
  "rf_gest",
  "rf_phyp",
  "rf_ghyp",
  "rf_eclam",
  "rf_ppterm",
  "rf_ppoutc",
  "rf_cesar",
  "rf_ncesar",
  "urf_diab",
  "urf_chyper",
  "urf_phyper",
  "urf_eclam",
  "op_cerv",
  "op_tocol",
  "op_ecvs",
  "op_ecvf",
  "uop_induc",
  "uop_tocol",
  "on_ruptr",
  "on_abrup",
  "on_prolg",
  "ld_induct",
  "ld_augment",
  "ld_nvrtx",
  "ld_steroids",
  "ld_antibio",
  "ld_chorio",
  "ld_mecon",
  "ld_fintol",
  "ld_anesth",
  "uld_meco",
  "uld_precip",
  "uld_breech",
  "md_attfor",
  "md_attvac",
  "md_present",
  "md_route",
  "md_trial",
  "ume_vag",
  "ume_vbac",
  "ume_primc",
  "ume_repec",
  "ume_forcp",
  "ume_vac",
  "rdmeth_rec",
  "udmeth_rec",
  "dmeth_rec",
  "attend",
  "apgar5",
  "apgar5r",
  "dplural",
  "imp_plur",
  "sex",
  "imp_sex",
  "dlmp_mm",
  "dlmp_dd",
  "dlmp_yy",
  "estgest",
  "combgest",
  "gestrec10",
  "gestrec3",
  "obgest_flg",
  "gest_imp",
  "dbwt",
  "bwtr14",
  "bwtr4",
  "bwtimp",
  "ab_vent",
  "ab_vent6",
  "ab_nicu",
  "ab_surfac",
  "ab_antibio",
  "ab_seiz",
  "ab_inj",
  "ca_anen",
  "ca_menin",
  "ca_heart",
  "ca_hernia",
  "ca_ompha",
  "ca_gastro",
  "ca_limb",
  "ca_cleftlp",
  "ca_cleft",
  "ca_downs",
  "ca_chrom",
  "ca_hypos",
  "uca_anen",
  "uca_spina",
  "uca_ompha",
  "uca_cleftlp",
  "uca_downs",
  "f_morigin",
  "f_forigin",
  "f_meduc",
  "f_clinest",
  "f_apgar5",
  "f_tobaco",
  "f_rf_pdiab",
  "f_rf_gdiab",
  "f_rf_phyper",
  "f_rf_ghyper",
  "f_rf_eclamp",
  "f_rf_ppb",
  "f_rf_ppo",
  "f_rf_cesar",
  "f_rf_ncesar",
  "f_ob_cervic",
  "f_ob_toco",
  "f_ob_succ",
  "f_ob_fail",
  "f_ol_rupture",
  "f_ol_precip",
  "f_ol_prolong",
  "f_ld_induct",
  "f_ld_augment",
  "f_ld_nvrtx",
  "f_ld_steroids",
  "f_ld_antibio",
  "f_ld_chorio",
  "f_ld_mecon",
  "f_ld_fintol",
  "f_ld_anesth",
  "f_md_attfor",
  "f_md_attvac",
  "f_md_present",
  "f_md_route",
  "f_md_trial",
  "f_ab_vent",
  "f_ab_vent6",
  "f_ab_nicu",
  "f_ab_surfac",
  "f_ab_antibio",
  "f_ab_seiz",
  "f_ab_inj",
  "f_ca_anen",
  "f_ca_menin",
  "f_ca_heart",
  "f_ca_hernia",
  "f_ca_ompha",
  "f_ca_gastro",
  "f_ca_limb",
  "f_ca_cleftlp",
  "f_ca_cleft",
  "f_ca_downs",
  "f_ca_chrom",
  "f_ca_hypos",
  "f_med",
  "f_wtgain",
  "f_tobac",
  "f_mpcb",
  "f_mpcb_u",
  "f_urf_diabetes",
  "f_urf_chyper",
  "f_urf_phyper",
  "f_urf_eclamp",
  "f_uob_induct",
  "f_uob_tocol",
  "f_uld_meconium",
  "f_uld_precip",
  "f_uld_breech",
  "f_u_vaginal",
  "f_u_vbac",
  "f_u_primac",
  "f_u_repeac",
  "f_u_forcep",
  "f_u_vacuum",
  "f_uca_anen",
  "f_uca_spina",
  "f_uca_omphalo",
  "f_uca_cleftlp",
  "f_uca_downs",
  "matchs",
  "aged",
  "ager5",
  "ager22",
  "manner",
  "dispo",
  "autopsy",
  "place",
  "ucod",
  "ucod130",
  "recwt",
]

X_COLUMNS = [
  # "dob_yy",
  # "dob_mm",
  # "dob_wk",
  # "ostate",
  # "ocntyfips",
  # "ocntypop",
  # "bfacil",
  # "ubfacil",
  # "bfacil3",
  # "mage_impflg",
  # "mage_repflg",
  "mager41",
  "mager14",
  "mager9",
  # "mbcntry",
  # "mrterr",
  # "mrcntyfips",
  # "rcnty_pop",
  # "rectype",
  # "restatus",
  # "mbrace",
  # "mrace",
  # "mracerec",
  # "mraceimp",
  # "umhisp",
  # "mracehisp",
  # "mar",
  # "mar_imp",
  # "meduc",
  # "umeduc",
  # "meduc_rec",
  # "fagerpt_flg",
  # "fagecomb",
  # "ufagecomb",
  # "fagerec11",
  # "fbrace",
  # "fracerec",
  # "ufhisp",
  # "fracehisp",
  # "frace",
  # "lbo",
  # "tbo",
  # "dllb_mm",
  # "dllb_yy",
  # "precare",
  # "precare_rec",
  # "mpcb",
  # "mpcb_rec6",
  # "mpcb_rec5",
  # "uprevis",
  # "previs_rec",
  "wtgain",
  "wtgain_rec",
  # "dfpc_imp",
  # "cig_1",
  # "cig_2",
  # "cig_3",
  # "tobuse",
  # "cigs",
  # "cig_rec6",
  # "cig_rec",
  # "rf_diab",
  # "rf_gest",
  # "rf_phyp",
  # "rf_ghyp",
  # "rf_eclam",
  # "rf_ppterm",
  # "rf_ppoutc",
  # "rf_cesar",
  # "rf_ncesar",
  # "urf_diab",
  # "urf_chyper",
  # "urf_phyper",
  # "urf_eclam",
  # "op_cerv",
  # "op_tocol",
  # "op_ecvs",
  # "op_ecvf",
  # "uop_induc",
  # "uop_tocol",
  # "on_ruptr",
  # "on_abrup",
  # "on_prolg",
  # "ld_induct",
  # "ld_augment",
  # "ld_nvrtx",
  # "ld_steroids",
  # "ld_antibio",
  # "ld_chorio",
  # "ld_mecon",
  # "ld_fintol",
  # "ld_anesth",
  # "uld_meco",
  # "uld_precip",
  # "uld_breech",
  # "md_attfor",
  # "md_attvac",
  # "md_present",
  # "md_route",
  # "md_trial",
  # "ume_vag",
  # "ume_vbac",
  # "ume_primc",
  # "ume_repec",
  # "ume_forcp",
  # "ume_vac",
  # "rdmeth_rec",
  # "udmeth_rec",
  # "dmeth_rec",
  # "attend",
  # "apgar5",
  # "apgar5r",
  # "dplural",
  # "imp_plur",
  # "sex",
  # "imp_sex",
  # "dlmp_mm",
  # "dlmp_dd",
  # "dlmp_yy",
  # "estgest",
  # "combgest",
  # "gestrec10",
  # "gestrec3",
  # "obgest_flg",
  # "gest_imp",
  # "dbwt",
  # "bwtr14",
  # "bwtr4",
  # "bwtimp",
  # "ab_vent",
  # "ab_vent6",
  # "ab_nicu",
  # "ab_surfac",
  # "ab_antibio",
  # "ab_seiz",
  # "ab_inj",
  # "ca_anen",
  # "ca_menin",
  # "ca_heart",
  # "ca_hernia",
  # "ca_ompha",
  # "ca_gastro",
  # "ca_limb",
  # "ca_cleftlp",
  # "ca_cleft",
  # "ca_downs",
  # "ca_chrom",
  # "ca_hypos",
  # "uca_anen",
  # "uca_spina",
  # "uca_ompha",
  # "uca_cleftlp",
  # "uca_downs",
  # "f_morigin",
  # "f_forigin",
  # "f_meduc",
  # "f_clinest",
  # "f_apgar5",
  "f_tobaco",
  "f_rf_pdiab",
  "f_rf_gdiab",
  # "f_rf_phyper",
  # "f_rf_ghyper",
  # "f_rf_eclamp",
  # "f_rf_ppb",
  # "f_rf_ppo",
  # "f_rf_cesar",
  # "f_rf_ncesar",
  # "f_ob_cervic",
  # "f_ob_toco",
  # "f_ob_succ",
  # "f_ob_fail",
  # "f_ol_rupture",
  # "f_ol_precip",
  # "f_ol_prolong",
  # "f_ld_induct",
  # "f_ld_augment",
  # "f_ld_nvrtx",
  # "f_ld_steroids",
  # "f_ld_antibio",
  # "f_ld_chorio",
  # "f_ld_mecon",
  # "f_ld_fintol",
  # "f_ld_anesth",
  # "f_md_attfor",
  # "f_md_attvac",
  # "f_md_present",
  # "f_md_route",
  # "f_md_trial",
  # "f_ab_vent",
  # "f_ab_vent6",
  # "f_ab_nicu",
  # "f_ab_surfac",
  # "f_ab_antibio",
  # "f_ab_seiz",
  # "f_ab_inj",
  # "f_ca_anen",
  # "f_ca_menin",
  # "f_ca_heart",
  # "f_ca_hernia",
  # "f_ca_ompha",
  # "f_ca_gastro",
  # "f_ca_limb",
  # "f_ca_cleftlp",
  # "f_ca_cleft",
  # "f_ca_downs",
  # "f_ca_chrom",
  # "f_ca_hypos",
  # "f_med",
  # "f_wtgain",
  # "f_tobac",
  # "f_mpcb",
  # "f_mpcb_u",
  # "f_urf_diabetes",
  # "f_urf_chyper",
  # "f_urf_phyper",
  # "f_urf_eclamp",
  # "f_uob_induct",
  # "f_uob_tocol",
  # "f_uld_meconium",
  # "f_uld_precip",
  # "f_uld_breech",
  # "f_u_vaginal",
  # "f_u_vbac",
  # "f_u_primac",
  # "f_u_repeac",
  # "f_u_forcep",
  # "f_u_vacuum",
  # "f_uca_anen",
  # "f_uca_spina",
  # "f_uca_omphalo",
  # "f_uca_cleftlp",
  # "f_uca_downs",
  # "matchs",
  # "aged",
  # "ager5",
  # "ager22",
]

import pandas as pd

deaths = pd.read_csv("deaths_2010.csv", names=DEATH_COLUMNS, skiprows=1)
births = pd.read_csv("births_2010_24174.csv", names=BIRTH_COLUMNS, skiprows=1)
