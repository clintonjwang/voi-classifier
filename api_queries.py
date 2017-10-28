from dicom.examples import anonymize
import requests
import os
import time

def search_vna(user, pw, acc_num=None, study=None, series=None, region='test', limit=None, modality="MR"):
    if region == 'test':
        host = 'vnatest1vt'
        port = '8083'
    elif region == 'prod':
        host = '10.47.11.221'
        port = '8083'
    else:
        raise ValueError("Unsupported region")


    url = ''.join(['http://', host, ':', port,
                   "/AcuoREST/dicomrs/search/studies"])

    if acc_num is not None:
        url += "?AccessionNumber=" + acc_num

    elif study is not None:
        url += "/" + study + "/series"

        if series is not None:
            url += "/" + series + "/instances"
        elif modality is not None:
            url += "?Modality=" + modality

    if limit is not None:
        if "?" in url:
            url += "&"
        else:
            url += "?"
        url += "limit=" + str(limit)
    #url += "&includefield=all"

    r = requests.get(url, auth=(user, pw)) #, headers=headers
    #if r.status_code != 200:
        #raise ValueError("Invalid request (response code %d) for URL: %s" % (r.status_code, url))
        
    return r, url


def retrieve_vna(user, pw, filepath, study=None, series=None, instance=None, region='test', metadata=False):
    """If metadata is true, filepath should end in xml. Else end in dcm."""

    if region == 'test':
        host = 'vnatest1vt'
        port = '8083'
    elif region == 'prod':
        host = '10.47.11.221'
        port = '8083'
    else:
        raise ValueError("Unsupported region")

    if metadata:
        url = ''.join(['http://', host, ':', port,
                       "/AcuoREST/dicomrs/retrieve/studies/",
                        study])

        if series is not None:
            url += "/series/" + series
            if instance is not None:
                url += "/instances/" + instance

        url += "/metadata"+"?contentType=application/xml"

        r = requests.get(url, auth=(user, pw)) #, headers=headers

        if r.status_code != 200:
            print("Skipped series:", series)
            #raise ValueError("Invalid request (response code %d) for URL: %s" % (r.status_code, url))
            return None, url

        with open(filepath, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

    else:
        url = ''.join(['http://', host, ':', port,
                       "/AcuoREST/wadoget?requestType=WADO&contentType=application/dicom&studyUID=",
                        study])

        if series is not None:
            url += "&seriesUID=" + series
            if instance is not None:
                url += "&objectUID=" + instance

        r = requests.get(url, auth=(user, pw)) #, headers=headers

        if r.status_code != 200:
            #print("Skipped instance:", instance)
            #raise ValueError("Invalid request (response code %d) for URL: %s" % (r.status_code, url))
            return None, url

        save_dir = os.path.dirname(filepath)
        with open(save_dir + "\\temp.dcm", 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

        anonymize.anonymize(filename = save_dir + "\\temp.dcm", output_filename=filepath)

        os.remove(save_dir + "\\temp.dcm")
        
    return r, url


def query_accession_num(user, pw, acc_nums, target_dir, exclude_terms=[], region="prod", verbose=False):
    """Arguments: acc_nums should be a list of accession numbers (as strings).
    target_dir is the directory to save the images to.
    exclude_terms is a list of terms to look for to exclude irrelevant protocols.
    If verbose is True, prints to screen as each series is loaded. Otherwise, only prints as each study is loaded.

    Each accession number (study) is saved to a separate folder named with its study UID.
    Within that folder, each series is saved to a separate subfolder, named with the series description."""

    tot_time = time.time()
    for acc_num in acc_nums:
        r, url = search_vna(user, pw, region=region, acc_num=acc_num, )
        try:
            study = r.json()[0]['0020000D']['Value'][0]
        except:
            print('Accession number', acc_num, 'has no studies associated with it.')
            continue
        print('Loading accession number', acc_num)

        r, url = search_vna(user, pw, region=region, study=study, )
        study_info = r.json()
        series = list(set([ser['0020000E']['Value'][0] for ser in study_info]))

        instances = {}

        for ser in series:
            r, url = search_vna(user, pw, region=region, study=study, series=ser, modality=None)
            series_info = r.json()
            instances[ser] = [inst['00080018']['Value'][0] for inst in series_info]

        acc_num_dir = target_dir + "\\" + acc_num

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if os.path.exists(acc_num_dir):
            print(acc_num_dir, "may already have been downloaded (folder already exists in target directory). Skipping.")
            continue
        else:
            os.makedirs(acc_num_dir)

        total = 0
        skip_ser = 0
        skip_inst = 0
        rmdir = []

        t = time.time()
        for ser in instances:
            if verbose:
                print("\n==============")
                print("Loading metadata for series", ser)
            series_dir = acc_num_dir + "\\" + ser

            if not os.path.exists(series_dir):
                os.makedirs(series_dir)

            r, url = retrieve_vna(user, pw, region=region, filepath = series_dir+"\\metadata.xml",
                              study=study, series=ser, metadata=True)
            if r is None:
                skip_ser += 1
                if verbose:
                    print("Skipping series.")
                continue

            txt = r.text
            search = '<DicomAttribute tag="0008103E" vr="LO" keyword="SeriesDescription">\r\n      <Value number="1">'
            index = txt.find(search) + len(search)
            protocol_name = txt[index:index + txt[index:].find("</Value>")].lower()
            protocol_name = protocol_name.replace("/", "-")
            protocol_name = protocol_name.replace("\\", "-")
            protocol_name = protocol_name.replace(":", "-")
            protocol_name = protocol_name.replace("?", "")
            protocol_name = protocol_name.replace("*", "")
            
            while os.path.exists(acc_num_dir + "\\" + protocol_name):
                protocol_name += "+"
            os.rename(series_dir, acc_num_dir + "\\" + protocol_name)
            series_dir = acc_num_dir + "\\" + protocol_name


            skip = False
            for exc_keyword in exclude_terms:
                if exc_keyword in protocol_name:
                    skip_ser += 1
                    skip = True
                    if verbose:
                        print("Skipping images for series with description", protocol_name)
                    rmdir.append(protocol_name)
                    break

            if skip:
                continue

            if verbose:
                print("Loading images for series with description:", protocol_name)

            for count, inst in enumerate(instances[ser]):
                r, _ = retrieve_vna(user, pw, region=region, filepath = series_dir+"\\"+str(count)+".dcm",
                              study=study, series=ser, instance=inst)

                if r is not None:
                    skip_inst += 1
                if verbose:
                    print(".", end="")

            total += count

        if len(rmdir)>0 and not os.path.exists(acc_num_dir+"\\others"):
            os.makedirs(acc_num_dir+"\\others")

            for d in rmdir:
                os.rename(acc_num_dir + "\\" + d, acc_num_dir + "\\others\\" + d)

        if verbose:
            print("Series loaded: ", len(series)-skip_ser, "/", len(series), sep="")
            print("Total images loaded:", total)
            print("Images skipped:", skip_inst)
            print("\nTime elapsed: %.2fs" % (time.time()-t))
            
    print("\nTime elapsed: %.1fs" % (time.time()-tot_time))


def query_accession_num_clinton(user, pw, acc_nums, target_dir, exclude_terms=[], region="prod", verbose=False):
    """Arguments: acc_nums should be a list of accession numbers (as strings).
    target_dir is the directory to save the images to.
    exclude_terms is a list of terms to look for to exclude irrelevant protocols.
    If verbose is True, prints to screen as each series is loaded. Otherwise, only prints as each study is loaded."""

    tot_time = time.time()
    for acc_num in acc_nums:
        r, url = search_vna(user, pw, region=region, acc_num=acc_num)
        try:
            study = r.json()[0]['0020000D']['Value'][0]
        except:
            print('Accession number', acc_num, 'has no studies associated with it.')
            continue
        print('Loading accession number', acc_num)

        r, url = search_vna(user, pw, region=region, study=study)
        study_info = r.json()
        series = list(set([ser['0020000E']['Value'][0] for ser in study_info]))

        instances = {}

        for ser in series:
            r, url = search_vna(user, pw, region=region, study=study, series=ser)
            series_info = r.json()
            instances[ser] = [inst['00080018']['Value'][0] for inst in series_info]
            if len(instances[ser]) < 7:
                series.remove(ser)
                if verbose:
                    print("Skipping series", ser)
                del instances[ser]

        acc_num_dir = target_dir + "\\" + acc_num

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if os.path.exists(acc_num_dir):
            print(acc_num_dir, "may already have been downloaded (folder already exists in target directory). Skipping.")
            continue
        else:
            os.makedirs(acc_num_dir)

        total = 0
        skip_ser = 0
        skip_inst = 0
        rmdir = []

        t = time.time()
        for ser in instances:
            if verbose:
                print("\n==============")
                print("Loading metadata for series", ser)
            series_dir = acc_num_dir + "\\" + ser

            if not os.path.exists(series_dir):
                os.makedirs(series_dir)

            r, url = retrieve_vna(user, pw, region=region, filepath = series_dir+"\\metadata.xml",
                              study=study, series=ser, metadata=True)
            if r is None:
                skip_ser += 1
                if verbose:
                    print("Skipping series.")
                continue

            txt = r.text
            search = '<DicomAttribute tag="0008103E" vr="LO" keyword="SeriesDescription">\r\n      <Value number="1">'
            index = txt.find(search) + len(search)
            protocol_name = txt[index:index + txt[index:].find("</Value>")].lower()
            protocol_name = protocol_name.replace("/", "-")
            protocol_name = protocol_name.replace("\\", "-")
            protocol_name = protocol_name.replace(":", "-")
            protocol_name = protocol_name.replace("?", "")
            protocol_name = protocol_name.replace("*", "")
            
            search = '<DicomAttribute tag="00200011" vr="IS" keyword="SeriesNumber">\r\n      <Value number="1">'
            index = txt.find(search) + len(search)
            series_num = txt[index:index + txt[index:].find("</Value>")]
            
            if "pre" in protocol_name:
                protocol_name = "T1_BL"
            elif "art" in protocol_name:
                protocol_name = "T1_AP"
            elif "ven" in protocol_name or "portal" in protocol_name or "pv" in protocol_name:
                protocol_name = "T1_VP"
            elif "equ" in protocol_name:
                protocol_name = "T1_EQ"
            else:
                protocol_name += "_" + series_num
            
            while os.path.exists(acc_num_dir + "\\" + protocol_name):
                protocol_name += "+"
            os.rename(series_dir, acc_num_dir + "\\" + protocol_name)
            series_dir = acc_num_dir + "\\" + protocol_name


            skip = False
            for exc_keyword in exclude_terms:
                if exc_keyword in protocol_name:
                    skip_ser += 1
                    skip = True
                    if verbose:
                        print("Skipping images for series with description", protocol_name)
                    rmdir.append(protocol_name)
                    break

            if skip:
                continue

            if verbose:
                print("Loading images for series with description:", protocol_name)

            for count, inst in enumerate(instances[ser]):
                r, _ = retrieve_vna(user, pw, region=region, filepath = series_dir+"\\"+str(count)+".dcm",
                              study=study, series=ser, instance=inst)

                if r is not None:
                    skip_inst += 1
                if verbose:
                    print(".", end="")

            total += count

        if len(rmdir)>0 and not os.path.exists(acc_num_dir+"\\others"):
            os.makedirs(acc_num_dir+"\\others")
            
            for d in rmdir:
                os.rename(acc_num_dir + "\\" + d, acc_num_dir + "\\others\\" + d)

        if verbose:
            print("Series loaded: ", len(series)-skip_ser, "/", len(series), sep="")
            print("Total images loaded:", total)
            print("Images skipped:", skip_inst)
            print("\nTime elapsed: %.2fs" % (time.time()-t))
            
    print("\nTime elapsed: %.1fs" % (time.time()-tot_time))


def setup_ui(skip_col=False, skip_exc=True):
    """UI flow. Returns None if cancelled or terminated with error, else returns
    _____ to exclude."""
    if not easygui.msgbox(('This utility retrieves studies from the YNHH VNA given a list of accession numbers. '
                        'It saves each study to its own folder, with each series stored in a separate subfolder.\n'
                        'NOTE: If the program is terminated early and rerun, it will skip the last.\n'
                        'WARNING: This assumes that folders labeled with one MRN will not contain files with another MRN.\n'
                        'It assumes a folder is labeled with an MRN if it contains 5 or more digits in a row.')):
        return None

    user, pw = eg.multpasswordbox(msg='Enter credentials to access VNA.', fields=["Username", "Password"])
    if user is None or pw is None:
        return None

    acc_nums = eg.multenterbox(msg='Enter accession numbers separated by commas (e.g. 12345678, E123456789, E234567890).')
    if acc_nums is None:
        return None
    else:
        acc_nums = acc_nums.trim().split(',')

    search_path = easygui.diropenbox(msg='Select a folder to save your images to.')
    if search_path is None:
        return None

    return [user, pw, acc_nums]


def main():
    """Starting point for script"""
    # Ask user for inputs
    ret = setup_ui()
    if ret is None:
        return
    else:
        (user, pw, acc_nums, target_dir) = ret
        query_accession_num(user, pw, acc_nums, target_dir)

if __name__ == "__main__":
    main()